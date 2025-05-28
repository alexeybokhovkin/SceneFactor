import os
import json
import re
from tqdm.auto import tqdm
import numpy as np
import warnings
from collections import OrderedDict
from scipy.special import softmax

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
        # filter semantic maps from other files
        self.npy_files = [os.path.join(self.datadir, x) for x in os.listdir(self.datadir) if x.endswith('.npy') and 'geo' not in x]

        self.npy_files = sorted(self.npy_files)
        self.npy_files = [x for i, x in enumerate(self.npy_files) if i % num_proc == proc]

    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx):

        f = self.npy_files[idx]
        sem = np.load(f) # (10, 32, 16, 32)
        sem = softmax(sem, axis=0)
        sem = np.argmax(sem, axis=0)
        sem_vox = torch.LongTensor(sem)

        onehot_sem = F.one_hot(sem_vox, num_classes=10)
        onehot_sem = torch.permute(onehot_sem, (3, 0, 1, 2)).float()

        return onehot_sem, f
    

# path to the folder where output after inference_separate_sem.py is stored
DATA_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/recon_semantic_16sp_old_l2_med_qwen_finetune_it400k_450scenes_camready_separate'
# SAVE_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/stage1_model_separate'
# SAVE_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/old_model_separate'
SAVE_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/new_model_as_stage1'
# SAVE_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/new_model_as_old'


@torch.no_grad()
def test_generation(num_proc=1, proc=0, data_dir=None, save_dir=None):

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

        hidden_dims = [16, 32, 64, 128, 128]
        context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims).cuda()
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
        batch_size = 4
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=batch_size, shuffle=False)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
                vox, filename = data

                filename_tokens = filename[0].split('/')
                chunk_id = filename_tokens[-1].split('.')[0]
                prefix = '/'.join(filename_tokens[:-1])
                # if os.path.exists(os.path.join(prefix, f'{chunk_id}_geo_orig420k.npy')):
                #     continue

                vox = vox.cuda()
                condition = context_encoder(vox)

                all_geo_coords_filtered = []
                for sem_map in vox:
                    sem_map_numpy = sem_map.cpu().detach().numpy()
                    sem_map_numpy = softmax(sem_map_numpy, axis=0)
                    sem_map_numpy = np.argmax(sem_map_numpy, axis=0)
                    chunk_sem_floor = sem_map_numpy.max(axis=1)
                    non_trivial_coords = np.where(chunk_sem_floor != 0)
                    non_trivial_coords = np.vstack(non_trivial_coords).T
                    if len(non_trivial_coords) != 0:
                        min_coord = non_trivial_coords.min(axis=0) * 8
                        max_coord = (non_trivial_coords.max(axis=0) + 0.9) * 8
                        geo_coords = np.dstack(np.meshgrid(np.arange(256), np.arange(256))).reshape((-1, 2))
                        geo_coords_filtered = [x for x in geo_coords if not ((x[0] >= min_coord[0]) & (x[1] >= min_coord[1])) & ((x[0] <= max_coord[0]) & (x[1] <= max_coord[1]))]
                    else:
                        geo_coords_filtered = []
                    if len(geo_coords_filtered) != 0:
                        geo_coords_filtered = np.array(geo_coords_filtered)
                        all_geo_coords_filtered += [geo_coords_filtered]
                    else:
                        all_geo_coords_filtered += [[]]

                samples, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition, mode='geometry')

                recon = model.vae_model.decode(samples)
                for i in range(recon.shape[0]):
                    filename_tokens = filename[i].split('/')
                    chunk_id = filename_tokens[-1].split('.')[0]
                    prefix = '/'.join(filename_tokens[:-1])

                    recon_sample = recon[i].detach().cpu().numpy()

                    prefix = save_dir
                    np.save(os.path.join(prefix, f'{chunk_id}_geo_new_75k.npy'), recon_sample)
                    np.save(os.path.join(prefix, f'{chunk_id}_lat_new_75k.npy'), samples[i].cpu().detach().numpy())
                    np.save(os.path.join(prefix, f'{chunk_id}_cond_new_75k.npy'), condition[i].cpu().detach().numpy())
                    np.save(os.path.join(prefix, f'{chunk_id}_voxcond_new_75k.npy'), vox[i].cpu().detach().numpy())

    
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
    arg_parser.add_argument(
        "--save_dir", "-s", required=True,
        help="This is a directory to save results",
    )
    arg_parser.add_argument('-n', '--num_proc', default=1, type=int)
    arg_parser.add_argument('-p', '--proc', default=0, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    os.makedirs(args.save_dir, exist_ok=True)
    test_generation(args.num_proc, args.proc, args.data_dir, args.save_dir)