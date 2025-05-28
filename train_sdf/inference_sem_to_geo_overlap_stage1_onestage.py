import os
import json
import re
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


def second_stage_inference(num_proc=1, proc=0, save_dir=None, target_scenes_file=None):

    # with open("/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/qwen_scenes_final_camready.txt", 'r') as fin:
    #     target_scenes = fin.readlines()
    with open(target_scenes_file, 'r') as fin:
        target_scenes = fin.readlines()
    target_scenes = [x[:-1] for x in target_scenes]

    rooms_dir = specs["SemPath"]

    all_rooms = [x for x in target_scenes]
    all_rooms = [x for i, x in enumerate(all_rooms) if i % num_proc == proc]

    for scene_id in all_rooms:
        if os.path.exists(os.path.join(save_dir, scene_id, 'full_geo.npy')):
            continue
        if os.path.exists(os.path.join(rooms_dir, scene_id)):
            test_generation(os.path.join(rooms_dir, scene_id))


@torch.no_grad()
def test_generation(scene_path):

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

        hidden_dims = [16, 32, 64, 128, 128] # 288
        context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims, superres=False).cuda().eval()
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

    # old model
    # chunk_size = 64
    # sem_chunk_size = 16
    # latent_size = 16
    # feature_size = 1

    # stage_1 model
    chunk_size = 128 
    sem_chunk_size = 16
    latent_size = 16
    feature_size = 1

    scene_id = scene_path.split('/')[-1]

    try:
        full_sem_cond = np.load(os.path.join(scene_path, 'full_sem.npy'))

        full_sem_cond = softmax(full_sem_cond, axis=0)
        full_sem_cond = np.argmax(full_sem_cond, axis=0)

        full_sem_cond = torch.LongTensor(full_sem_cond)
        onehot_sem = F.one_hot(full_sem_cond, num_classes=10)
        onehot_sem = torch.permute(onehot_sem, (3, 0, 1, 2)).float()[None, ...]
    except FileNotFoundError:
        return

    num_x_chunks = onehot_sem.shape[2] // sem_chunk_size - 1
    num_y_chunks = onehot_sem.shape[4] // sem_chunk_size - 1
    if num_x_chunks < 1:
        num_x_chunks = 1
    if num_y_chunks < 1:
        num_y_chunks = 1

    full_room_recon = torch.zeros((chunk_size * (num_x_chunks + 1), chunk_size, chunk_size * (num_y_chunks + 1))).float()
    full_room_latent = torch.zeros((feature_size, latent_size * (num_x_chunks + 1), latent_size, latent_size * (num_y_chunks + 1))).float()
    full_room_processed = torch.zeros((feature_size, latent_size * (num_x_chunks + 1), latent_size, latent_size * (num_y_chunks + 1))).long()

    SAVE_DIR = os.path.join(args.save_dir, scene_id)

    for i_x in range(num_x_chunks):
        for i_y in range(num_y_chunks):

            sem_chunk = onehot_sem[:, :, i_x * sem_chunk_size: (i_x + 2) * sem_chunk_size, :, i_y * sem_chunk_size: (i_y + 2) * sem_chunk_size].cuda()
            condition = context_encoder(sem_chunk)

            latent_init = full_room_latent[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)]
            latent_mask = full_room_processed[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)]
            latent_init = latent_init.cuda()
            latent_mask = latent_mask.cuda()

            samples, _, _ = model.diffusion_model.generate_conditional_room(1, cond=condition, latent_init=latent_init, latent_mask=latent_mask, traj=False, mode='geometry')

            recon = model.vae_model.decode(samples)

            latent_init = latent_init.cpu()
            latent_mask = latent_mask.cpu()
            samples = samples.cpu()
            full_room_latent[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)] = \
                samples[0].cpu().detach()
            full_room_processed[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)] = 1


    for i_x in range(num_x_chunks):
        for i_y in range(num_y_chunks):
            latent_chunk = full_room_latent[:, i_x * latent_size: (i_x + 2) * latent_size, :, i_y * latent_size: (i_y + 2) * latent_size].cuda()
            recon = model.vae_model.decode(latent_chunk[None, ...])
            full_room_recon[i_x * chunk_size: (i_x + 2) * chunk_size, :, i_y * chunk_size: (i_y + 2) * chunk_size] = recon[0, 0]
    
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
    arg_parser.add_argument(
        "--target_scenes_file", "-t", required=True,
        help="Path to a .txt file with scene ids to process",
    )
    arg_parser.add_argument('-n', '--num_proc', default=1, type=int)
    arg_parser.add_argument('-p', '--proc', default=0, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    os.makedirs(args.save_dir, exist_ok=True)
    second_stage_inference(args.num_proc, args.proc, args.save_dir, args.target_scenes_file)
        

  
