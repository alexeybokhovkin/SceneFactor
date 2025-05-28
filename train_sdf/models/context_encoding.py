import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

from models.x_transformer import Encoder, TransformerWrapper


class ContextEncoder(nn.Module):

    def __init__(self,
                 hidden_dims: List = None,
                 superres=False,
                 **kwargs) -> None:
        super(ContextEncoder, self).__init__()

        self.hidden_dims = hidden_dims

        enc_modules = []
        if not superres:
            enc_input_channels = 10
        else:
            enc_input_channels = 1 # superres
        for h_dim in hidden_dims[:1]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        for h_dim in hidden_dims[1:2]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1) if not superres else nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        for h_dim in hidden_dims[2:]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        self.encoder = nn.Sequential(*enc_modules)

    def forward(self, x):

        for layer in self.encoder:
            x = layer(x)

        return x
    

class TextEncoder(nn.Module):

    def __init__(self,
                 in_channels: 512,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(TextEncoder, self).__init__()

        self.hidden_dims = hidden_dims

        layer_1 = nn.Sequential(
                    nn.Linear(in_channels, self.hidden_dims[0]),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                    nn.LeakyReLU()
                  )
        layer_2 = nn.Sequential(
                    nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
                    nn.BatchNorm1d(self.hidden_dims[1]),
                    nn.LeakyReLU()
                  )
        layer_3 = nn.Sequential(
                    nn.Linear(self.hidden_dims[1], self.hidden_dims[2]),
                    nn.BatchNorm1d(self.hidden_dims[2]),
                    nn.LeakyReLU()
                  )
        layer_4 = nn.Sequential(
                    nn.Linear(self.hidden_dims[2], self.hidden_dims[3]),
                    nn.BatchNorm1d(self.hidden_dims[3]),
                    nn.LeakyReLU()
                  )
        layers = [layer_1, layer_2, layer_3, layer_4]
        self.encoder = nn.Sequential(*layers)


    def forward(self, x):

        all_samples = []
        for k in range(x.shape[1]):
            x_sample = x[:, k, :]
            for layer in self.encoder:
                x_sample = layer(x_sample)
            all_samples += [x_sample[:, None]]
        all_samples = torch.cat(all_samples, dim=1)

        return all_samples
    

class Text3DEncoder(nn.Module):

    def __init__(self,
                 in_channels: 8,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Text3DEncoder, self).__init__()

        self.hidden_dims = hidden_dims
    
        # Full conv encoding ([1, 10, 8, 8, 8] input) [16, 32, 64, 128, 288]
        enc_modules = []
        enc_input_channels = in_channels
        for h_dim in hidden_dims[:1]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=1, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        for h_dim in hidden_dims[1:2]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        for h_dim in hidden_dims[2:]:
            enc_modules.append(
                nn.Sequential(
                    nn.Conv3d(enc_input_channels, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            enc_input_channels = h_dim
        self.encoder = nn.Sequential(*enc_modules)

    def forward(self, x):

        for layer in self.encoder:
            x = layer(x)

        return x
    

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c
    

class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast 
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return self.tokenizer.convert_ids_to_tokens(text)


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text) #.to(self.device)
        else:
            tokens = text
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)