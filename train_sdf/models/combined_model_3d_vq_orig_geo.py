import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl

# add paths in model/__init__.py for new models
from models import * 
from models.openai_model import UNetModel
from models.context_encoding import BERTEmbedder


class CombinedModel3DVQOrigGeo(pl.LightningModule):
    def __init__(self, specs, exp_dir=None):
        super().__init__()
        self.specs = specs
        self.exp_dir = exp_dir

        self.task = specs['training_task']

        vae_specs = specs['ddconfig']
        n_embed = specs['n_embed']
        embed_dim = specs['embed_dim']
        ae_version = specs['ae_version'] if 'ae_version' in specs else False
        self.ae_version = ae_version

        self.vae_model = VQVAE(vae_specs, n_embed, embed_dim, ae_version=ae_version)

        self.nll_loss = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if self.task in ('combined', 'diffusion'): 
            self.diffusion_model = GaussianDiffusion(model=UNetModel(**specs["diffusion_model_specs"]), **specs["diffusion_specs"])
            self.context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device='cuda')


    def accuracy_w_back(self, pred_sem, gt_sem):
        pred_sem_exp = torch.argmax(pred_sem, dim=1)
        gt_sem_exp = torch.argmax(gt_sem, dim=1)
        acc_map = (pred_sem_exp == gt_sem_exp).float()
        acc_map = torch.mean(acc_map, dim=[1, 2, 3])
        acc_map_batch = torch.mean(acc_map)
        return acc_map_batch
    
    def iou_w_back(self, pred_sem, gt_sem):

        gt_sem_sum = torch.sum(gt_sem, dim=[2, 3, 4])

        intersection = pred_sem * gt_sem # [B, C, H, W, D]
        union = pred_sem + gt_sem - intersection
        iou_bc = torch.sum(intersection, dim=[2, 3, 4]) / (torch.sum(union, dim=[2, 3, 4]) + 1e-6)
        iou_bc[torch.where(gt_sem_sum == 0)] = 1
        iou_c = torch.mean(iou_bc, dim=0)
        iou = torch.mean(iou_bc[torch.where(gt_sem_sum != 0)])

        return iou, iou_c


    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x, idx)
        elif self.task == 'modulation':
            return self.train_modulation(x, idx)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        
    def validation_step(self, x, idx):

        if self.task == 'combined':
            return self.val_combined(x, idx)
        elif self.task == 'modulation':
            return self.val_modulation(x, idx)
        elif self.task == 'diffusion':
            return self.val_diffusion(x)
        

    def configure_optimizers(self):


        params_list = [
                {'params': self.parameters(), 'lr': self.specs['sdf_lr'], 'betas': (0.5, 0.9)}
            ]
       
        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.85) # gamma=0.7
                }
        }


    def train_modulation(self, x, idx):

        vox = x['gt_vox']
        scale = 1

        vox = vox * scale

        out = self.vae_model(vox, verbose=False)
        pred_geo, emb_loss = out[0], out[1]
        emb_loss = emb_loss.mean()

        sdf_loss = 1 * F.l1_loss(pred_geo.squeeze(), vox.squeeze(), reduction='none')
        sdf_loss = sdf_loss.mean()

        w_emb = 0 if self.ae_version else 1
        loss = sdf_loss + w_emb * emb_loss

        loss_dict = {"sdf": sdf_loss, "emb": emb_loss}
        loss_dict_agg = {"sdf_agg": sdf_loss, "emb_agg": emb_loss}

        print('Loss dict', loss_dict)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict_agg, prog_bar=False, enable_graph=False, on_epoch=True, sync_dist=True)

        return loss
    

    def val_modulation(self, x, idx):

        vox = x['gt_vox']
        scale = 1

        vox = vox * scale

        out = self.vae_model(vox, verbose=False)
        pred_geo, emb_loss = out[0], out[1]
        emb_loss = emb_loss.mean()

        sdf_loss = 1 * F.l1_loss(pred_geo.squeeze(), vox.squeeze(), reduction='none')
        sdf_loss = sdf_loss.mean()

        loss_dict = {"val_sdf": sdf_loss, "val_emb": emb_loss}
        loss_dict_agg = {"val_sdf_agg": sdf_loss, "val_emb_agg": emb_loss}

        print('Loss dict', loss_dict)
        self.log_dict(loss_dict_agg, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True, on_step=True)

    def train_combined(self, x, idx):

        vox = x['gt_vox']
        condition = x['caption']

        latent, emb_loss, _ = self.vae_model.encode(vox)
        pred_sem = self.vae_model.decode(latent)
        emb_loss = emb_loss.mean()

        gt_sem_exp = torch.argmax(vox, dim=1)
        pred_sem_nll = self.log_softmax(pred_sem)
        sem_loss = self.nll_loss(pred_sem_nll, gt_sem_exp).mean()

        acc_map_batch = self.accuracy_w_back(torch.exp(pred_sem_nll), vox) # only semantic, takes 10 channels with background
        iou, iou_c = self.iou_w_back(torch.exp(pred_sem_nll), vox)
        print('IoU', iou_c)

        condition = list(condition)
        condition = self.context_encoder(condition)
        condition[np.where(np.random.rand(condition.shape[0]) < 0.1)] = 0

        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=condition)
        
        pred_sem_diff = self.vae_model.decode(pred_latent)
        pred_sem_diff_nll = self.log_softmax(pred_sem_diff)
        sem_diff_loss = self.nll_loss(pred_sem_diff_nll, gt_sem_exp).mean()

        acc_map_batch_diff = self.accuracy_w_back(torch.exp(pred_sem_diff_nll), vox) # only semantic, takes 10 channels with background
        iou_diff, iou_c_diff = self.iou_w_back(torch.exp(pred_sem_diff_nll), vox)
        print('IoU diff', iou_c_diff)

        loss = sem_loss + emb_loss + diff_loss + sem_diff_loss


        loss_dict = {"sem": sem_loss, 
                     "emb": emb_loss, 
                     "diff": diff_loss,
                     "sem_diff": sem_diff_loss,
                     "acc": acc_map_batch, 
                     "iou": iou,
                     "acc_diff": acc_map_batch_diff,
                     "iou_diff": iou_diff}
        loss_dict_agg = {"sem_agg": sem_loss, 
                         "emb_agg": emb_loss, 
                         "diff_agg": diff_loss,
                         "sem_diff_agg": sem_diff_loss,
                         "acc_agg": acc_map_batch, 
                         "iou_agg": iou,
                         "acc_diff_agg": acc_map_batch_diff,
                         "iou_diff_agg": iou_diff}

        print('Loss dict', loss_dict)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict_agg, prog_bar=False, enable_graph=False, on_epoch=True, sync_dist=True)

        return loss
    
    def val_combined(self, x, idx):

        vox = x['gt_vox']
        condition = x['caption']

        latent, emb_loss, _ = self.vae_model.encode(vox)
        pred_sem = self.vae_model.decode(latent)
        emb_loss = emb_loss.mean()

        gt_sem_exp = torch.argmax(vox, dim=1)
        pred_sem_nll = self.log_softmax(pred_sem)
        sem_loss = self.nll_loss(pred_sem_nll, gt_sem_exp).mean()

        acc_map_batch = self.accuracy_w_back(torch.exp(pred_sem_nll), vox) # only semantic, takes 10 channels with background
        iou, iou_c = self.iou_w_back(torch.exp(pred_sem_nll), vox)
        print('IoU', iou_c)

        condition = list(condition)
        condition = self.context_encoder(condition)
        condition[np.where(np.random.rand(condition.shape[0]) < 0.1)] = 0

        diff_loss, diff_100_loss, diff_1000_loss, pred_latent, perturbed_pc = self.diffusion_model.diffusion_model_from_latent(latent, cond=condition)
        
        pred_sem_diff = self.vae_model.decode(pred_latent)
        pred_sem_diff_nll = self.log_softmax(pred_sem_diff)
        sem_diff_loss = self.nll_loss(pred_sem_diff_nll, gt_sem_exp).mean()

        acc_map_batch_diff = self.accuracy_w_back(torch.exp(pred_sem_diff_nll), vox) # only semantic, takes 10 channels with background
        iou_diff, iou_c_diff = self.iou_w_back(torch.exp(pred_sem_diff_nll), vox)
        print('IoU diff', iou_c_diff)


        loss_dict = {"sem": sem_loss, 
                     "emb": emb_loss, 
                     "diff": diff_loss,
                     "sem_diff": sem_diff_loss,
                     "acc": acc_map_batch, 
                     "iou": iou,
                     "acc_diff": acc_map_batch_diff,
                     "iou_diff": iou_diff}
        loss_dict_agg = {"sem_agg": sem_loss, 
                         "emb_agg": emb_loss, 
                         "diff_agg": diff_loss,
                         "sem_diff_agg": sem_diff_loss,
                         "acc_agg": acc_map_batch, 
                         "iou_agg": iou,
                         "acc_diff_agg": acc_map_batch_diff,
                         "iou_diff_agg": iou_diff}

        print('Loss dict', loss_dict)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict_agg, prog_bar=False, enable_graph=False, on_epoch=True, sync_dist=True)


   
