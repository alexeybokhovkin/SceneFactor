import os
import json
import re
from tqdm.auto import tqdm
import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.utils.data 
from torch.nn import functional as F

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import * 
from models import ContextEncoder



class DatasetGeo(torch.utils.data.Dataset):

    def __init__(
        self,
        datadir=None,
        num_proc=1,
        proc=0
    ):
 
        self.datadir = datadir
        # filter geometric maps from other files
        self.npy_files = [os.path.join(self.datadir, x) for x in os.listdir(self.datadir) if x.endswith('.npy') and 'geo' in x]

        self.npy_files = sorted(self.npy_files)
        self.npy_files = [x for i, x in enumerate(self.npy_files) if i % num_proc == proc]

        # get the list of corresponding latent filenames
        self.latent_files = []
        for npy_file in self.npy_files:
            tokens = npy_file.split('geo_big600k.npy')
            self.latent_files += [tokens[0] + 'geolatent_big600k.npy']

    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):

        f = self.npy_files[idx]

        f_latent = self.latent_files[idx]
        latent = np.load(f_latent)

        return latent, f


DATA_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/recon_semantic_16sp_old_l2_med_qwen_finetune_it400k_450scenes_camready_separate'


@torch.no_grad()
def test_generation(num_proc=1, proc=0, data_dir=None):

    print('Start generation')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start loading checkpoint')
        model = CombinedModel3DVQOrigGeo.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)

        ckpt = torch.load(specs["diffusion_ckpt_path"])

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in ckpt['model_state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = ckpt['model_state_dict']

        model.diffusion_model.load_state_dict(model_dict)
        model = model.cuda().eval()
        print('Finish loading checkpoint')

        hidden_dims = [16, 32, 64, 64, 128]
        context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims, superres=True).cuda()
        context_ckpt = torch.load(specs["diffusion_ckpt_path_context"])

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in context_ckpt['model_state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = context_ckpt['model_state_dict']

        context_encoder.load_state_dict(model_dict)
        context_encoder = context_encoder.cuda().eval()

        test_dataset = DatasetGeo(data_dir, num_proc, proc)
        batch_size = 1
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=batch_size, shuffle=False)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
                latent, filename = data

                latent_cond_1 = latent[:, :, :16, :, :16].cuda()
                latent_cond_2 = latent[:, :, 16:, :, :16].cuda()
                latent_cond_3 = latent[:, :, :16, :, 16:].cuda()
                latent_cond_4 = latent[:, :, 16:, :, 16:].cuda()

                condition_1 = context_encoder(latent_cond_1)
                condition_2 = context_encoder(latent_cond_2)
                condition_3 = context_encoder(latent_cond_3)
                condition_4 = context_encoder(latent_cond_4)

                samples_1, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition_1)
                samples_2, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition_2)
                samples_3, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition_3)
                samples_4, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition_4)

                full_recon_latent = torch.zeros((1, 1, 64, 32, 64)).cuda()
                full_recon_latent[:, :, :32, :, :32] = samples_1
                full_recon_latent[:, :, 32:, :, :32] = samples_2
                full_recon_latent[:, :, :32, :, 32:] = samples_3
                full_recon_latent[:, :, 32:, :, 32:] = samples_4

                full_recon = model.vae_model.decode(full_recon_latent)
                for i in range(full_recon.shape[0]):
                    filename_tokens = filename[i].split('/')
                    chunk_id = filename_tokens[-1].split('.')[0]
                    prefix = '/'.join(filename_tokens[:-1])

                    np.save(os.path.join(prefix, f'{chunk_id}_geo_600k_highreslat375k.npy'), full_recon[i].detach().cpu().numpy())
                    np.save(os.path.join(prefix, f'{chunk_id}_geolatent_600k_highreslat375k.npy'), full_recon_latent[i].detach().cpu().numpy())

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--data_dir", "-d", required=True,
        help="This directory should contain separate semantic chunks generated in the previous step with inference_separate_sem.py",
    )
    arg_parser.add_argument('-n', '--num_proc', default=1, type=int)
    arg_parser.add_argument('-p', '--proc', default=0, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    test_generation(args.num_proc, args.proc, args.data_dir)