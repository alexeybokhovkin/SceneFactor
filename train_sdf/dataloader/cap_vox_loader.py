import os
import random

import torch
import torch.utils.data
import numpy as np
import json
import torch.nn.functional as F

from . import base 


class CapVoxloader(base.Dataset):

    def __init__(
        self,
        split_file,
        return_filename=False,
        rot_index=None,
        chunk_size=64, # 64/128 for geo, 16 for sem
        big_chunk_size=180,
        sampled_latent='geo',
        orig_chunks_dir=None
    ):

        self.return_filename = return_filename
        self.gt_paths = self.get_instance_filenames_scene_vox_list(split_file)
        self.point_clouds = self.gt_paths
        self.near_surface_sdf = 0.07
        self.sampled_latent = sampled_latent
        self.chunk_size = chunk_size
        self.big_chunk_size = big_chunk_size

        self.rot_index = rot_index

        if orig_chunks_dir is None:
            self.orig_chunks_dir = '/cluster/daidalos/abokhovkin/Front3D/chunked_data'
        else:
            self.orig_chunks_dir = orig_chunks_dir

    def get_all_files(self):
        return self.point_clouds, self.gt_paths 
    
    def __getitem__(self, idx):

        if self.sampled_latent == 'sem':
            vox = self.sample_for_latent_sem(self.gt_paths[idx])
        else:
            vox = self.sample_for_latent_geo(self.gt_paths[idx])
        if self.return_filename:
            return vox, self.gt_paths[idx], idx
        else:
            return vox, idx


    def __len__(self):
        return len(self.gt_paths)

    def sample_for_latent_geo(self, f): 

        f_tokens = f.split('/')
        f_folder = '/'.join(f_tokens[:-1])
        scene_id = f_tokens[-2]

        try:
            json_filename = f_tokens[-1].split('.')[0] + '.json'
            with open(os.path.join(f_folder, json_filename)) as fin:
                meta_data = json.load(fin)
            
            subchunk_ids = meta_data['chunk_ids']
            subchunk_ids = list(set(subchunk_ids))
            subchunk_bounds = meta_data['subchunk_bounds_simple']
            subchunk_ids_x = [int(x.split('_')[0]) for x in subchunk_ids]
            subchunk_ids_y = [int(x.split('_')[2]) for x in subchunk_ids]
            min_id_x, min_id_y = np.min(subchunk_ids_x), np.min(subchunk_ids_y)
            big_vox = np.zeros((self.big_chunk_size * 2, self.big_chunk_size, self.big_chunk_size * 2))
            if os.path.exists(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x}_0_{min_id_y}.npy')):
                c00 = np.load(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x}_0_{min_id_y}.npy'))
                big_vox[:self.big_chunk_size, :, :self.big_chunk_size] = c00
            if os.path.exists(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x + 1}_0_{min_id_y}.npy')):
                c10 = np.load(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x + 1}_0_{min_id_y}.npy'))
                big_vox[self.big_chunk_size:, :, :self.big_chunk_size] = c10
            if os.path.exists(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x}_0_{min_id_y + 1}.npy')):
                c01 = np.load(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x}_0_{min_id_y + 1}.npy'))
                big_vox[:self.big_chunk_size, :, self.big_chunk_size:] = c01
            if os.path.exists(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x + 1}_0_{min_id_y + 1}.npy')):
                c11 = np.load(os.path.join(self.orig_chunks_dir, scene_id, f'{min_id_x + 1}_0_{min_id_y + 1}.npy'))
                big_vox[self.big_chunk_size:, :, self.big_chunk_size:] = c11
            
            gt_vox = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                             subchunk_bounds[0][1]:subchunk_bounds[1][1],
                             subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            assert gt_vox.shape == (2 * self.chunk_size, self.chunk_size, 2 * self.chunk_size)

        except ValueError as e:
            print('Error', e)
            gt_vox = np.ones((2 * self.chunk_size, self.chunk_size, 2 * self.chunk_size))
        
        
        if self.rot_index is None:
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
        else:
            p_rot = self.rot_index
        gt_vox = torch.FloatTensor(gt_vox)
        gt_vox = torch.rot90(gt_vox, k=p_rot, dims=[0, 2])
        gt_vox = gt_vox.numpy()
        
        gt_vox = np.abs(gt_vox)
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox[gt_vox < -self.near_surface_sdf] = -self.near_surface_sdf

        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        num_channels = gt_vox.shape[0]
        if gt_vox.shape != (num_channels, self.chunk_size * 2, self.chunk_size, self.chunk_size * 2):
            gt_buf = torch.ones((num_channels, self.chunk_size * 2, self.chunk_size, self.chunk_size * 2)) * self.near_surface_sdf
            if num_channels > 1:
                gt_buf[:, :, :, :] = 0
                gt_buf[:, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[:, :, :, :]
                gt_vox = gt_buf

        # gt_vox = gt_vox[:, ::4, ::4, ::4]

        return gt_vox
    

    def sample_for_latent_sem(self, f): 

        # chunk_size = 16
        f_tokens = f.split('/')
        f_folder = '/'.join(f_tokens[:-1])

        try:
            vox_filename = f_tokens[-1].split('.')[0] + '_semantic.npy'
            gt_vox = np.load(os.path.join(f_folder, vox_filename))
        except ValueError as e:
            print('Error', e)
            gt_vox = np.ones((2 * self.chunk_size, self.chunk_size, 2 * self.chunk_size))
        
        if self.rot_index is None:
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
        else:
            p_rot = self.rot_index
        gt_vox = torch.FloatTensor(gt_vox)
        gt_vox = torch.rot90(gt_vox, k=p_rot, dims=[0, 2])
        gt_vox = gt_vox.numpy()
        
        gt_vox = torch.LongTensor(gt_vox)
        gt_vox = F.one_hot(gt_vox, num_classes=10)
        gt_vox = torch.permute(gt_vox, (3, 0, 1, 2)).float()

        num_channels = gt_vox.shape[0]
        if gt_vox.shape != (num_channels, self.chunk_size * 2, self.chunk_size, self.chunk_size * 2):
            gt_buf = torch.zeros((num_channels, self.chunk_size * 2, self.chunk_size, self.chunk_size * 2))
            if num_channels > 1:
                gt_buf[:, :, :, :] = 0
                gt_buf[:, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[:, :, :, :]
                gt_vox = gt_buf

        # gt_vox = gt_vox[:, ::4, ::4, ::4]

        return gt_vox

   
class Roomloader(base.Dataset):

    def __init__(
        self,
        room_path,
        caption_key,
        chunk_size=64,
        double_chunks=True,
        caption_type='qwen'
    ):

        self.near_surface_sdf = 0.07
        self.room_path = room_path
        self.chunk_size = chunk_size
        self.double_chunks = double_chunks
        self.caption_type = caption_type

        self.all_room_chunks = sorted([x.split('.')[0] for x in os.listdir(room_path) if x.endswith('.json')])
        self.max_x_chunk = np.max([int(filename.split('_')[0]) for filename in self.all_room_chunks])
        self.max_y_chunk = np.max([int(filename.split('_')[2]) for filename in self.all_room_chunks])

        self.all_room_chunks = [(int(x.split('_')[0]), int(x.split('_')[2])) for x in self.all_room_chunks]
        self.all_room_chunks = sorted(self.all_room_chunks, key=lambda x: 100 * x[0] + x[1])

        self.caption_key = caption_key

    def get_room_size(self):
        return self.max_x_chunk, self.max_y_chunk
    
    def __getitem__(self, idx): 
        vox, caption = self.get_chunk(self.all_room_chunks[idx])
        return vox, self.all_room_chunks[idx], idx, caption

    def __len__(self):
        return len(self.all_room_chunks)

    def get_chunk(self, chunk_tokens): 

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        chunk_filename = f'{chunk_tokens[0]}_0_{chunk_tokens[1]}.npy'
        f = os.path.join(self.room_path, chunk_filename)

        try:
            gt_vox = np.load(f)[:, :self.chunk_size, :]
        except ValueError:
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
        
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        if gt_vox.shape != (1, chunk_size_eff, self.chunk_size, chunk_size_eff):
            gt_buf = torch.ones((1, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[:, :, :, :]
            gt_vox = gt_buf

        meta_data_path = f.split('.')[0] + '.json'
        with open(meta_data_path, 'r') as fin:
            meta_data = json.load(fin)

        captions_layout_fixed = {
            "top left corner": "with walls positioned at the top and left",
            "top right corner": "with walls positioned at the top and right",
            "bottom left corner": "with walls positioned at the bottom and left",
            "bottom right corner": "with walls positioned at the bottom and right",
            "wall on the top": "with walls positioned at the top",
            "wall on the bottom": "with walls positioned at the bottom",
            "wall on the left": "with walls positioned at the left",
            "wall on the right": "with walls positioned at the right",
            "": ""
        }

        if self.caption_type != 'qwen':
            # synthetic caption
            random.seed(13)
            caption_list = meta_data[self.caption_key]

            if isinstance(caption_list, list):
                random.shuffle(caption_list)
                caption = ''
                sep = ', '
                for item in caption_list:
                    caption = caption + item + sep
                caption = caption[:-2]
            else:
                caption = caption_list
            caption_wall = captions_layout_fixed[caption_wall]

            if not isinstance(caption_list, list):
                if self.caption_key not in ['caption_spatial', 'caption_subcat_spatial']:
                    caption_list = caption_list.split(', ')
                else:
                    caption_list = caption_list.split('; ')
            random.shuffle(caption_list)
            caption = ''
            sep = ', '
            for item in caption_list:
                caption = caption + item + sep
            caption = caption[:-2]

            caption = caption_list
            
            caption_walls = meta_data['caption_layout_manual']
            caption_wall = caption_walls[0]
            caption_wall = captions_layout_fixed[caption_wall]
            
            if caption.endswith('.'):
                caption = caption[:-1]
            if caption_wall != '':
                caption = caption + ', ' + caption_wall


        else:
            if 'full_qwen_caption' in meta_data:
                caption = meta_data['full_qwen_caption']
                caption_walls = meta_data['caption_layout']
                p_rot = 0
                caption_wall = caption_walls[p_rot]
                if caption_wall != '':
                    caption = caption + ', ' + caption_wall
            else:
                print('Qwen caption was not found, using original')
                random.seed(13)
                caption_list = meta_data[self.caption_key]
                if not isinstance(caption_list, list):
                    if self.caption_key not in ['caption_spatial', 'caption_subcat_spatial']:
                        caption_list = caption_list.split(', ')
                    else:
                        caption_list = caption_list.split('; ')
                random.shuffle(caption_list)
                caption = ''
                sep = ', '
                for item in caption_list:
                    caption = caption + item + sep
                caption = caption[:-2]

                caption_walls = meta_data['caption_layout']
                # caption_walls = meta_data['caption_wall']
                p_rot = 0
                caption_wall = caption_walls[p_rot]
                if caption_wall != '':
                    caption = caption + ', ' + caption_wall

        return gt_vox, caption
