# adopt from: 
# - VQVAE: https://github.com/nadavbh12/VQ-VAE
# - Encoder: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py

from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class VQVAE_Encoder(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim
                 ):
        super(VQVAE_Encoder, self).__init__()

        self.ddconfig = ddconfig
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.encoder = Encoder3D(**ddconfig)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)

        self.mlp = nn.Sequential(*[nn.Linear(256, 128, bias=True),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128, bias=True),
                                   nn.ReLU(),
                                   ])

        init_weights(self.encoder, 'normal', 0.02)
        init_weights(self.quant_conv, 'normal', 0.02)
        init_weights(self.mlp, 'normal', 0.02)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return quant, emb_loss, info

    def encode_no_quant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        b, f, h_, w_, d_ = h.shape
        h = h.reshape((b * f, -1))
        h = self.mlp(h)
        return h

    def forward(self, input, forward_no_quant=False):

        if forward_no_quant:
            z = self.encode_no_quant(input)
            diff = 0
            return z, diff

        quant, diff, info = self.encode(input)
        quant = self.post_quant_conv(quant)

        return quant, diff
    

class MLPDecoder(nn.Module):
    def __init__(self, in_feat_dims, out_channels, use_b_norm,
                 non_linearity=nn.ReLU(inplace=True), closure=None):
        super(MLPDecoder, self).__init__()

        previous_feat_dim = in_feat_dims
        all_ops = []

        for depth in range(len(out_channels)):
            out_dim = out_channels[depth]
            affine_op = nn.Linear(previous_feat_dim, out_dim, bias=True)
            all_ops.append(affine_op)

            if depth < len(out_channels) - 1:
                if use_b_norm:
                    all_ops.append(nn.BatchNorm1d(out_dim))

                if non_linearity is not None:
                    all_ops.append(non_linearity)

            previous_feat_dim = out_dim

        if closure is not None:
            all_ops.append(closure)

        self.net = nn.Sequential(*all_ops)

    def forward(self, x):
        return self.net(x)
    

class LanguageEncoder(nn.Module):
    """
    Currently it reads the tokens via an LSTM initialized on a specific context feature and
    return the last output of the LSTM.
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    """
    def __init__(self, n_hidden, embedding_dim, vocab_size, padding_idx=0):
        super(LanguageEncoder, self).__init__()
        # Whenever the embedding sees the padding index
        # it'll make the whole vector zeros
        self.padding_idx = padding_idx
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    embedding_dim=embedding_dim,
        #                                    padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, n_hidden, batch_first=True)

    def forward(self, padded_tokens, init_feats=None, drop_out_rate=0.3):
        w_emb = F.dropout(padded_tokens, drop_out_rate, self.training)
        len_of_sequence = (padded_tokens != self.padding_idx).sum(dim=1)[:, 0].cpu()
        x_packed = pack_padded_sequence(w_emb, len_of_sequence, enforce_sorted=False, batch_first=True)

        context_size = 1
        if init_feats is not None:
            context_size = init_feats.shape[1]

        batch_size = len(padded_tokens)
        res = []
        for i in range(context_size):
            init_i = init_feats[:, i].contiguous()
            init_i = torch.unsqueeze(init_i, 0)    # rep-mat if multiple LSTM cells.
            rnn_out_i, _ = self.rnn(x_packed, (init_i, init_i))
            rnn_out_i, dummy = pad_packed_sequence(rnn_out_i, batch_first=True)
            lang_feat_i = rnn_out_i[torch.arange(batch_size), len_of_sequence - 1]
            res.append(lang_feat_i)
        return res
    

def smoothed_cross_entropy(pred, target, alpha=0.1):
    n_class = pred.size(1)
    # one_hot = torch.zeros_like(pred).scatter_(1, target.view(-1, 1), 1)
    one_hot = target
    one_hot = one_hot * ((1.0 - alpha) + alpha / n_class) + (1.0 - one_hot) * alpha / n_class  # smoothed
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return torch.mean(loss)