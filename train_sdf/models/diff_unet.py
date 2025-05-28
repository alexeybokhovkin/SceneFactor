
import numpy as np
import torch
import torch.utils.data 
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl

from models.openai_model import UNetModel
from models.diffusion import GaussianDiffusion
from models.context_encoding import ContextEncoder, BERTEmbedder


class DiffusionUNet(pl.LightningModule):
    def __init__(self, config, exp_dir=None, train_set=None, val_set=None, cond_mode=None):
        super().__init__()
        self.config = config
        self.exp_dir = exp_dir

        self.train_set = train_set
        self.val_set = val_set
        self.cond_mode = cond_mode

        self.model = GaussianDiffusion(
            model=UNetModel(**config['model_specs']),
            **config['diffusion_specs']
        )

        if 'text' in cond_mode:
            self.context_encoder = BERTEmbedder(n_embed=1280, n_layer=32)
        elif 'sem' in cond_mode:
            self.hidden_dims = [16, 32, 64, 128, 128] # 288
            self.context_encoder = ContextEncoder(in_channels=1, hidden_dims=self.hidden_dims)

        self.loss_100, self.loss_500, self.loss_1000  = [1], [1], [1]

    def configure_optimizers(self):
    
        if 'text' in self.cond_mode:
            params_list = [
                    {'params': self.model.parameters(), 'lr': self.config['diff_lr']},
                    {'params': self.context_encoder.parameters(), 'lr': self.config['diff_lr']}
                ]
        elif 'sem' in self.cond_mode:
            params_list = [
                    {'params': self.model.parameters(), 'lr': self.config['diff_lr']},
                    {'params': self.context_encoder.parameters(), 'lr': 5e-5}
                ]
       
        optimizer = torch.optim.AdamW(params_list)
        opt = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
            }
        }
        return opt

    def forward(self, x, idx):
        data, condition = x
        t = torch.randint(0, self.model.num_timesteps, (self.config['batch_size'],), device=data.device).long()

        if 'text' in self.cond_mode:
            condition = list(condition)
            condition = self.context_encoder(condition)
        elif 'sem' in self.cond_mode:
            condition = condition.cuda()
            condition = self.context_encoder(condition)

        condition[np.where(np.random.rand(condition.shape[0]) < 0.06)] = 0

        # unconditional 
        loss, xt, target, pred, unreduced_loss, loss_dict = self.model(data, t, ret_pred_x=True, cond=condition)

        return loss, unreduced_loss, loss_dict, t.cpu()

    def training_step(self, x, idx):

        loss, unreduced_loss, loss_logs, t = self.forward(x, idx)

        self.loss_100.extend([x.mean() for x in unreduced_loss[t<100].cpu().numpy()])
        self.loss_500.extend([x.mean() for x in unreduced_loss[t<500].cpu().numpy()])
        self.loss_1000.extend([x.mean() for x in unreduced_loss[t>500].cpu().numpy()])

        logs = {
            "train_loss": loss_logs["loss"],
            "unet_loss": loss_logs["loss_simple"],
            "vlb_loss": loss_logs["loss_vlb"],
        }

        if self.global_step % self.config['loss_freq'] == 0:
            logs["loss_100"] = np.mean(self.loss_100)
            logs["loss_500"] = np.mean(self.loss_500)
            logs["loss_1000"] = np.mean(self.loss_1000)
            self.loss_100, self.loss_500, self.loss_1000 = [0], [0], [0]

        self.log_dict(logs, prog_bar=True, enable_graph=False, sync_dist=True, on_step=True, logger=True)

        batch_dictionary = {
            "loss": loss,
            "train_loss": loss_logs["loss"],
            "unet_loss": loss_logs["loss_simple"],
            "vlb_loss": loss_logs["loss_vlb"],
        }

        return batch_dictionary

    # @rank_zero_only
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        avg_unet_loss = torch.stack([x["unet_loss"] for x in outputs]).mean()
        avg_vlb_loss = torch.stack([x["vlb_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "avg_train_loss": avg_train_loss,
            "avg_unet_loss": avg_unet_loss,
            "avg_vlb_loss": avg_vlb_loss,
        }

        for output in outputs:
            all_loss_100, all_loss_500, all_loss_1000 = [], [], []
            if "loss_100" in output:
                all_loss_100 += [output["loss_100"]]
                all_loss_500 += [output["loss_500"]]
                all_loss_1000 += [output["loss_1000"]]
        if len(all_loss_100) > 0:
            avg_loss_100 = torch.stack(all_loss_100).mean()
            avg_loss_500 = torch.stack(all_loss_500).mean()
            avg_loss_1000 = torch.stack(all_loss_1000).mean()
            tensorboard_logs["avg_loss_100"] = avg_loss_100
            tensorboard_logs["avg_loss_500"] = avg_loss_500
            tensorboard_logs["avg_loss_1000"] = avg_loss_1000

        self.log_dict(tensorboard_logs, logger=True, prog_bar=True)

    def validation_step(self, x, idx):
        loss, unreduced_loss, loss_logs, t = self.forward(x, idx)

        self.loss_100.extend([x.mean() for x in unreduced_loss[t<100].cpu().numpy()])
        self.loss_500.extend([x.mean() for x in unreduced_loss[t<500].cpu().numpy()])
        self.loss_1000.extend([x.mean() for x in unreduced_loss[t>500].cpu().numpy()])

        logs = {
            "val_loss": loss_logs["loss"],
            "val_unet_loss": loss_logs["loss_simple"],
            "val_vlb_loss": loss_logs["loss_vlb"],
        }

        if self.global_step % 1 == 0:
            logs["val_loss_100"] = np.mean(self.loss_100)
            logs["val_loss_500"] = np.mean(self.loss_500)
            logs["val_loss_1000"] = np.mean(self.loss_1000)
            self.loss_100, self.loss_500, self.loss_1000 = [0], [0], [0]

        self.log_dict(logs, prog_bar=True, enable_graph=False, sync_dist=True, on_step=True, logger=True)

        batch_dictionary = {
            "val_loss": loss_logs["loss"],
            "val_unet_loss": loss_logs["loss_simple"],
            "val_vlb_loss": loss_logs["loss_vlb"],
        }

        return batch_dictionary

    # @rank_zero_only
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_unet_loss = torch.stack([x["val_unet_loss"] for x in outputs]).mean()
        avg_val_vlb_loss = torch.stack([x["val_vlb_loss"] for x in outputs]).mean()
            
        tensorboard_logs = {
            "avg_val_loss": avg_val_loss,
            "avg_unet_loss": avg_val_unet_loss,
            "avg_vlb_loss": avg_val_vlb_loss,
        }

        self.log_dict(tensorboard_logs, logger=True, prog_bar=True)
