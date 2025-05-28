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

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *
from models import BERTEmbedder
from train_sdf.dataloader.test_loader import Roomloader



def first_stage_inference(num_proc=1, proc=0, scene_cap=None, save_dir=None):

    rooms_dir = specs["RoomsPath"]
    # with open('/cluster/balar/abokhovkin/data/Front3D/val_scenes_450_main_qwen_camready.json', 'r') as fin: # qwen
    #     val_scenes = json.load(fin)
    with open(scene_cap, 'r') as fin:
        val_scenes = json.load(fin)

    all_rooms = sorted(list(val_scenes.keys()))[:]
    all_rooms = [x for i, x in enumerate(all_rooms) if i % num_proc == proc]

    for scene_id in all_rooms:
        caption_key = val_scenes[scene_id]
        test_generation(os.path.join(rooms_dir, scene_id), caption_key, save_dir)


@torch.no_grad()
def test_generation(room_path, caption_key, save_dir=None):

    print('Start generation')
    factor = specs["Factor"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start loading checkpoint')
        if factor == 'sem':
            model = CombinedModel3DVQOrig.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
        else:
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

        context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device=f'cuda').cuda() # 1280 / 512
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

    filename_tokens = room_path.split('/')
    scene_id = filename_tokens[-1]

    test_dataset = Roomloader(room_path=room_path, caption_key=caption_key)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)

    max_room_x, max_room_y = test_dataset.get_room_size()
    room_size_x = max_room_x + 1
    room_size_y = max_room_y + 1

    latent_size = 16
    feature_dim = 1
    sem_dim = 10
    recon_size = 32

    full_room_latent = torch.zeros((feature_dim, latent_size // 2 * (room_size_x + 1), latent_size, latent_size // 2 * (room_size_y + 1))).float()
    full_room_processed = torch.zeros((feature_dim, latent_size // 2 * (room_size_x + 1), latent_size, latent_size // 2 * (room_size_y + 1))).long()
    full_room_processed_sem = torch.zeros((recon_size // 2 * (room_size_x + 1), recon_size // 2 * (room_size_y + 1))).long()

    all_captions = {}
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            vox, chunk_tokens, gt_idx, caption = data
            chunk_tokens = (int(chunk_tokens[0]), int(chunk_tokens[1]))
            all_captions[chunk_tokens] = caption[0]

            caption = list(caption)
            condition = context_encoder(caption)

            latent_coords = (latent_size // 2 * chunk_tokens[0], latent_size // 2 * chunk_tokens[1])
            latent_init = full_room_latent[:, latent_coords[0]:latent_coords[0] + latent_size, :, latent_coords[1]:latent_coords[1] + latent_size]
            latent_mask = full_room_processed[:, latent_coords[0]:latent_coords[0] + latent_size, :, latent_coords[1]:latent_coords[1] + latent_size]
            latent_init = latent_init.cuda()
            latent_mask = latent_mask.cuda()

            samples, _, _ = model.diffusion_model.generate_conditional_room(1, cond=condition, latent_init=latent_init, latent_mask=latent_mask, traj=False)

            latent_init = latent_init.cpu()
            latent_mask = latent_mask.cpu()
            samples = samples.cpu()
            full_room_latent[:, latent_coords[0]:latent_coords[0] + latent_size, :, latent_coords[1]:latent_coords[1] + latent_size] = \
                samples[0].cpu().detach()
            full_room_processed[:, latent_coords[0]:latent_coords[0] + latent_size, :, latent_coords[1]:latent_coords[1] + latent_size] = 1
            full_room_processed_sem[2 * latent_coords[0]:2 * (latent_coords[0] + latent_size), 2 * latent_coords[1]:2 * (latent_coords[1] + latent_size)] = 1


    SAVE_DIR = os.path.join(save_dir, scene_id)
    os.makedirs(SAVE_DIR, exist_ok=True)

    full_scene_recon = torch.zeros((sem_dim, recon_size // 2 * (room_size_x + 1), recon_size // 2, recon_size // 2 * (room_size_y + 1))).float()
    
    full_room_size_x = room_size_x
    full_room_size_y = room_size_y
    for chunk_id_x in range(full_room_size_x):
        for chunk_id_y in range(full_room_size_y):

            latent = full_room_latent[:, latent_size // 2 * chunk_id_x: latent_size // 2 * (chunk_id_x + 2), :, latent_size // 2 * chunk_id_y: latent_size // 2 * (chunk_id_y + 2)]
            latent = latent.cuda()

            occ_mask = full_room_processed_sem[latent_size * chunk_id_x: latent_size * (chunk_id_x + 2), latent_size * chunk_id_y: latent_size * (chunk_id_y + 2)]
            non_occ_coords = torch.where(occ_mask == 0)
            non_occ_coords = torch.stack(non_occ_coords).T

            sem_recon = model.vae_model.decode(latent[None, :, :, :8, :])

            chunk_sem = sem_recon.cpu().detach().numpy()[0]
            chunk_sem = softmax(chunk_sem, axis=0)
            chunk_sem = np.argmax(chunk_sem, axis=0)
            trivial_coords = np.where(chunk_sem == 0)
            trivial_coords = np.vstack(trivial_coords).T
            non_trivial_coords = np.where(chunk_sem != 0)
            non_trivial_coords = np.vstack(non_trivial_coords).T
            min_z_coord = non_trivial_coords[:, 1].min()
            bottom_trivial_coords = trivial_coords[trivial_coords[:, 1] == min_z_coord][:, [0, 2]]
            sem_recon[:, 1, bottom_trivial_coords[:, 0], min_z_coord, bottom_trivial_coords[:, 1]] = 1000.0
            sem_recon[:, 0, non_occ_coords[:, 0], :, non_occ_coords[:, 1]] = 1000.0
            sem_recon[:, 1, non_occ_coords[:, 0], :, non_occ_coords[:, 1]] = 0.0

            np.save(os.path.join(SAVE_DIR, "{}_{}_sem.npy".format(chunk_id_x, chunk_id_y)), sem_recon.cpu().numpy())

            chunk_coords = (recon_size // 2 * chunk_id_x, recon_size // 2 * chunk_id_y)
            full_scene_recon[:, chunk_coords[0]:chunk_coords[0] + recon_size, :, chunk_coords[1]:chunk_coords[1] + recon_size] = sem_recon[0]
    
            meta_data = {}
            if (chunk_id_x, chunk_id_y) in all_captions:
                meta_data['caption'] = all_captions[(chunk_id_x, chunk_id_y)]
                with open(os.path.join(SAVE_DIR, "{}_{}_recon.json".format(chunk_id_x, chunk_id_y)), 'w') as fout:
                    json.dump(meta_data, fout)

    np.save(os.path.join(SAVE_DIR, "full_sem.npy"), full_scene_recon.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "full_latent.npy"), full_room_latent.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "full_occ.npy"), full_room_processed_sem.cpu().numpy())

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--scene_cap", "-c", type=str,
        help="caption specs (per scene) to be used in generation process"
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
    # recon_dir = os.path.join(MAIN_SAVE_DIR, f"recon_semantic_16sp_old_l2_med_qwen_finetune_it400k_450scenes_camready")
    # os.makedirs(recon_dir, exist_ok=True)
    
    first_stage_inference(args.num_proc, args.proc, args.scene_cap, args.save_dir)
        

  
