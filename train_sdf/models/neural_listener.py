import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import time

# add paths in model/__init__.py for new models
from models import *
from models.context_encoding import BERTEmbedder

class Listener(pl.LightningModule):
    def __init__(self, specs, exp_dir=None):
        super().__init__()
        self.specs = specs
        self.exp_dir = exp_dir

        vae_specs = specs['ddconfig']
        n_embed = specs['n_embed']
        embed_dim = specs['embed_dim']

        n_hidden = specs['n_hidden']
        embedding_dim = specs['embedding_dim']
        vocab_size = specs['vocab_size']

        self.geo_encoder = VQVAE_Encoder(vae_specs, n_embed, embed_dim)
        self.text_encoder = BERTEmbedder(n_embed=128, n_layer=4, vocab_size=32768, max_seq_len=50)
        self.mlp_decoder = MLPDecoder(256, [128, 1], True)
        self.language_encoder = LanguageEncoder(n_hidden, embedding_dim, vocab_size)

        self.reg_gamma = 0.005

    def training_step(self, x, idx):

        return self.train_modulation(x, idx)
        
    def validation_step(self, x, idx):

        return self.val_modulation(x, idx)

    def configure_optimizers(self):


        params_list = [
                {'params': self.parameters(), 'lr': self.specs['sdf_lr']}
            ]
       
        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
        }
    
    def forward(self, voxes, captions):

        b, f, h, w, d = voxes.shape
        voxes_flat = voxes.reshape((b * f, 1, h, w, d))

        out = self.geo_encoder.encode_no_quant(voxes_flat)
        geo_latent = out
        geo_latent = geo_latent.reshape(b, f, -1)

        captions = list(captions)
        bert_embedding = self.text_encoder(captions)
        lang_feats = self.language_encoder(bert_embedding, init_feats=geo_latent)

        logits = []
        for i, l_feats in enumerate(lang_feats):
            feats = torch.cat([l_feats, geo_latent[:, i]], 1)

            logits.append(self.mlp_decoder(feats))
        all_logits = torch.cat(logits, 1)

        return all_logits


    def train_modulation(self, x, idx):

        voxes = x['gt_voxes']
        caption = x['caption']
        true_answer = x['true_answer']

        b, f, h, w, d = voxes.shape
        voxes_flat = voxes.reshape((b * f, 1, h, w, d))

        out = self.geo_encoder.encode_no_quant(voxes_flat)
        geo_latent = out
        geo_latent = geo_latent.reshape(b, f, -1)

        caption = list(caption)
        bert_embedding = self.text_encoder(caption)
        lang_feats = self.language_encoder(bert_embedding, init_feats=geo_latent)

        logits = []
        for i, l_feats in enumerate(lang_feats):
            feats = torch.cat([l_feats, geo_latent[:, i]], 1)

            logits.append(self.mlp_decoder(feats))
        all_logits = torch.cat(logits, 1)

        loss = smoothed_cross_entropy(all_logits, true_answer)                            
        reg_loss = 0.0
        
        for p in self.geo_encoder.named_parameters():
            if 'fc.weight' == p[0]:
                reg_loss += p[1].norm(2)
                
        reg_loss *= self.reg_gamma                
        full_loss = loss + reg_loss

        _, preds = torch.max(all_logits, 1)
        _, true_explicit = torch.max(true_answer, 1)
        running_corrects = torch.sum(preds == true_explicit)
        print('Accuracy', running_corrects.double() / b)

        loss_dict = {"sce": loss, "reg": reg_loss}
        loss_dict_agg = {"sce_agg": loss, "reg_agg": reg_loss}

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict_agg, prog_bar=False, enable_graph=False, on_epoch=True, sync_dist=True)

        return loss
    

    def val_modulation(self, x, idx):

        voxes = x['gt_voxes']
        caption = x['caption']
        true_answer = x['true_answer']

        b, f, h, w, d = voxes.shape
        voxes_flat = voxes.reshape((b * f, 1, h, w, d))

        out = self.geo_encoder.encode_no_quant(voxes_flat)
        geo_latent = out
        geo_latent = geo_latent.reshape(b, f, -1)

        caption = list(caption)
        bert_embedding = self.text_encoder(caption)
        lang_feats = self.language_encoder(bert_embedding, init_feats=geo_latent)

        logits = []
        for i, l_feats in enumerate(lang_feats):
            feats = torch.cat([l_feats, geo_latent[:, i]], 1)

            logits.append(self.mlp_decoder(feats))
        all_logits = torch.cat(logits, 1)

        loss = smoothed_cross_entropy(all_logits, true_answer)                            
        reg_loss = 0.0
        
        for p in self.geo_encoder.named_parameters():
            if 'fc.weight' == p[0]:
                reg_loss += p[1].norm(2)
                
        reg_loss *= self.reg_gamma     

        _, preds = torch.max(all_logits, 1)
        _, true_explicit = torch.max(true_answer, 1)
        running_corrects = torch.sum(preds == true_explicit)
        print('Accuracy', running_corrects.double() / b)

        loss_dict = {"val_sce": loss, "val_reg": reg_loss}
        loss_dict_agg = {"val_sce_agg": loss, "val_reg_agg": reg_loss}

        self.log_dict(loss_dict_agg, prog_bar=True, enable_graph=False, sync_dist=True)
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False, sync_dist=True, on_step=True)
