import os
import json
import re
import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.utils.data 
from torch.nn import functional as F

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *


def second_stage_inference(edit_scenes_folder=None):

    # SCENES_DIR = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/editing'
    SCENES_DIR = edit_scenes_folder
    all_scenes = [
        # '3f8e0c63-02e1-4a9a-962d-6a0b5d6e45d1_size_1',
        # '3f8e0c63-02e1-4a9a-962d-6a0b5d6e45d1_removal_1',
        # '6d0f311f-5d41-4f24-9484-eb44915ede82_addition_1',
        # '5b19af6a-1fd9-4903-96d5-f36433bd1fd6_replacement_1',
        # 'd42a800f-ad34-4c25-92c5-a1103a256fe8_move_1',

        # '3f8e0c63-02e1-4a9a-962d-6a0b5d6e45d1_size_2',
        # '3f8e0c63-02e1-4a9a-962d-6a0b5d6e45d1_removal_2',
        # '6d0f311f-5d41-4f24-9484-eb44915ede82_addition_2',
        # '5b19af6a-1fd9-4903-96d5-f36433bd1fd6_replacement_2',
        'd42a800f-ad34-4c25-92c5-a1103a256fe8_move_2',
    ]

    for scene_id in all_scenes:
        test_generation(os.path.join(SCENES_DIR, scene_id))


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


    chunk_size = 128
    sem_chunk_size = 16
    latent_size = 16
    feature_size = 1


    full_sem_cond = np.load(os.path.join(scene_path, 'full_sem.npy')) # original semantic scene
    full_sem_cond = torch.LongTensor(full_sem_cond)
    onehot_sem = F.one_hot(full_sem_cond, num_classes=10)
    onehot_sem = torch.permute(onehot_sem, (3, 0, 1, 2)).float()[None, ...]

    full_sem_cond_edited = np.load(os.path.join(scene_path, f'full_sem_edited.npy')) # edited semantic scene
    onehot_sem_edited = torch.LongTensor(full_sem_cond_edited)
    onehot_sem_edited = F.one_hot(onehot_sem_edited, num_classes=10)
    onehot_sem_edited = torch.permute(onehot_sem_edited, (3, 0, 1, 2)).float()[None, ...]

    with open(os.path.join(scene_path, f'full_sem_edited.json'), 'r') as fin: # edit operation annotation
        editions = json.load(fin)

    init_latent = np.load(os.path.join(scene_path, 'full_lowres_latent.npy')) # original scene geometric 3D latent grid
    init_latent = torch.FloatTensor(init_latent)


    num_x_chunks = onehot_sem.shape[2] // sem_chunk_size - 1
    num_y_chunks = onehot_sem.shape[4] // sem_chunk_size - 1
    if num_x_chunks < 1:
        num_x_chunks = 1
    if num_y_chunks < 1:
        num_y_chunks = 1


    full_room_recon = torch.zeros((chunk_size * (num_x_chunks + 1), chunk_size, chunk_size * (num_y_chunks + 1))).float()
    full_room_latent = torch.zeros((feature_size, latent_size * (num_x_chunks + 1), latent_size, latent_size * (num_y_chunks + 1))).float()
    full_room_latent[:, :, :, :] = init_latent
    full_room_processed = torch.ones((feature_size, latent_size * (num_x_chunks + 1), latent_size, latent_size * (num_y_chunks + 1))).long()

    for edit_key in editions:
        for edit_coords in editions[edit_key]:

            if edit_key != 'move':
                edit_coords = np.array(edit_coords)
                min_point = edit_coords.min(axis=0)
                max_point = edit_coords.max(axis=0)

                min_point = np.floor(min_point).astype('int16')
                max_point = np.ceil(max_point).astype('int16')

                min_point = min_point - 2
                max_point = max_point + 2
                min_point = np.clip(min_point, [0, 0, 0], [full_room_latent.shape[1]-1, full_room_latent.shape[2]-1, full_room_latent.shape[3]-1])
                max_point = np.clip(max_point, [0, 0, 0], [full_room_latent.shape[1]-1, full_room_latent.shape[2]-1, full_room_latent.shape[3]-1])

                full_room_latent[:, min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]] = 0
                full_room_processed[:, min_point[0]:max_point[0], min_point[1]:max_point[1], min_point[2]:max_point[2]] = 0
                keep_mask = torch.where(full_room_processed == 0, 0, 1)
                edit_mask = 1 - keep_mask

            else:
                edit_coords_0 = np.array(edit_coords[0])
                edit_coords_1 = np.array(edit_coords[1])
                min_point_0 = edit_coords_0.min(axis=0)
                max_point_0 = edit_coords_0.max(axis=0)
                min_point_1 = edit_coords_1.min(axis=0)
                max_point_1 = edit_coords_1.max(axis=0)

                min_point_0 = np.floor(min_point_0).astype('int16')
                max_point_0 = np.ceil(max_point_0).astype('int16')
                min_point_1 = np.floor(min_point_1).astype('int16')
                max_point_1 = np.ceil(max_point_1).astype('int16')

                interval_1 = max_point_1 - min_point_1
                # target location
                full_room_latent[:, min_point_0[0]:min_point_0[0] + interval_1[0], min_point_0[1]:min_point_0[1] + interval_1[1], min_point_0[2]:min_point_0[2] + interval_1[2]] = full_room_latent[:, min_point_1[0]:max_point_1[0], min_point_1[1]:max_point_1[1], min_point_1[2]:max_point_1[2]]
                full_room_processed[:, min_point_0[0]:min_point_0[0] + interval_1[0], min_point_0[1]:min_point_0[1] + interval_1[1], min_point_0[2]:min_point_0[2] + interval_1[2]] = 1 # 0 / 1
                
                # initial location 
                min_point_1 = min_point_1 - 1
                max_point_1 = max_point_1 + 1
                min_point_1 = np.clip(min_point_1, [0, 0, 0], [full_room_latent.shape[1]-1, full_room_latent.shape[2]-1, full_room_latent.shape[3]-1])
                max_point_1 = np.clip(max_point_1, [0, 0, 0], [full_room_latent.shape[1]-1, full_room_latent.shape[2]-1, full_room_latent.shape[3]-1])
                full_room_latent[:, min_point_1[0]:max_point_1[0], min_point_1[1]:max_point_1[1], min_point_1[2]:max_point_1[2]] = 0
                full_room_processed[:, min_point_1[0]:max_point_1[0], min_point_1[1]:max_point_1[1], min_point_1[2]:max_point_1[2]] = 0

                keep_mask = torch.where(full_room_processed == 0, 0, 1)
                edit_mask = 1 - keep_mask

    SAVE_DIR = scene_path

    for i_x in range(num_x_chunks):
        for i_y in range(num_y_chunks):

            sem_chunk = onehot_sem_edited[:, :, i_x * sem_chunk_size: (i_x + 2) * sem_chunk_size, :, i_y * sem_chunk_size: (i_y + 2) * sem_chunk_size].cuda()
            condition = context_encoder(sem_chunk)

            latent_init = full_room_latent[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)]
            latent_mask = full_room_processed[:, i_x * (latent_size): (i_x + 2) * (latent_size), :, i_y * (latent_size): (i_y + 2) * (latent_size)]
        
            if 0 not in latent_mask:
                continue

            latent_init = latent_init.cuda()
            latent_mask = latent_mask.cuda()

            samples, _, _ = model.diffusion_model.generate_conditional_room(1, cond=condition, latent_init=latent_init, latent_mask=latent_mask, traj=False)

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
    
    np.save(os.path.join(SAVE_DIR, f"full_geo_lowres_edited_{args.edit_num}.npy"), full_room_recon.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, f"full_room_latent_lowres_edited_{args.edit_num}.npy"), full_room_latent.cpu().numpy())


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--edit_scenes_folder", "-d", required=True,
        help="The directory with scenes to edit, where every scene contains orig. sem. map, edited sem. map, editing annotation, orig. geom. lat. grid",
    )
    arg_parser.add_argument(
        "--edit_num", "-k", default=None, type=str,
        help="Edit num",
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    
    second_stage_inference(args.edit_scenes_folder)
        

  
