import os
import json
import re
import numpy as np
import warnings
from collections import OrderedDict
import glob

import torch
import torch.utils.data 
from torch.nn import functional as F

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *
from models import ContextEncoder


def second_stage_inference(num_proc=1, proc=0, save_dir=None):

    rooms_dir = specs["SemPath"]
    with open('/cluster/balar/abokhovkin/data/Front3D/val_scenes_450_main_qwen_camready.json', 'r') as fin:
        val_scenes = json.load(fin)
    all_rooms = sorted(list(val_scenes.keys()))[:]
    
    all_rooms = [x for i, x in enumerate(all_rooms) if i % num_proc == proc]

    for scene_id in all_rooms:
        if os.path.exists(os.path.join(rooms_dir, scene_id)):
            test_generation(os.path.join(rooms_dir, scene_id), save_dir)


@torch.no_grad()
def test_generation(scene_path, save_dir=None):

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

        hidden_dims = [16, 32, 64, 64, 128]
        context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims, superres=True).cuda().eval()
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
        print('Finish loading checkpoint')


    chunk_size = 128
    sem_chunk_size = 16
    latent_size = 32
    feature_size = 1

    scene_id = scene_path.split('/')[-1]

    try:
        full_geo_cond = np.load(os.path.join(scene_path, 'full_room_latent.npy'))
        full_geo_cond = torch.FloatTensor(full_geo_cond)[None, ...]
    except FileNotFoundError:
        return

    num_x_chunks = full_geo_cond.shape[2] // sem_chunk_size
    num_y_chunks = full_geo_cond.shape[4] // sem_chunk_size
    if num_x_chunks < 1:
        num_x_chunks = 1
    if num_y_chunks < 1:
        num_y_chunks = 1

    full_room_recon = torch.zeros((chunk_size * (num_x_chunks), chunk_size, chunk_size * (num_y_chunks))).float()
    full_room_latent = torch.zeros((feature_size, latent_size * (num_x_chunks), latent_size, latent_size * (num_y_chunks))).float()

    SAVE_DIR = os.path.join(save_dir, scene_id)

    for i_x in range(num_x_chunks):
        for i_y in range(num_y_chunks):

            sem_chunk = full_geo_cond[:, :, i_x * sem_chunk_size: (i_x + 1) * sem_chunk_size, :, i_y * sem_chunk_size: (i_y + 1) * sem_chunk_size].cuda()
            condition = context_encoder(sem_chunk)

            samples, _ = model.diffusion_model.generate_conditional(1, cond=condition)
            recon = model.vae_model.decode(samples)

            samples = samples.cpu()
            full_room_latent[:, i_x * (latent_size): (i_x + 1) * (latent_size), :, i_y * (latent_size): (i_y + 1) * (latent_size)] = \
                samples[0].cpu().detach()

    for i_x in range(num_x_chunks):
        for i_y in range(num_y_chunks):
            latent_chunk = full_room_latent[:, i_x * latent_size: (i_x + 1) * latent_size, :, i_y * latent_size: (i_y + 1) * latent_size].cuda()
            recon = model.vae_model.decode(latent_chunk[None, ...])
            full_room_recon[i_x * chunk_size: (i_x + 1) * chunk_size, :, i_y * chunk_size: (i_y + 1) * chunk_size] = recon[0, 0]
    
    np.save(os.path.join(SAVE_DIR, "full_geo.npy"), full_room_recon.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "full_room_latent.npy"), full_room_latent.cpu().numpy())


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--save_dir", "-s", required=True,
        help="This is a directory to save results",
    )
    arg_parser.add_argument('-n', '--num_proc', default=1, type=int)
    arg_parser.add_argument('-p', '--proc', default=0, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    # MAIN_SAVE_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2'
    # recon_dir = os.path.join(MAIN_SAVE_DIR, f"recon_geo_it375k_qwen_superres_450scenes_camready")
    # os.makedirs(recon_dir, exist_ok=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    second_stage_inference(args.num_proc, args.proc, args.save_dir)
        

  
