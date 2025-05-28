import os
import random

import numpy as np
import json
import torch
import torch.utils.data

from . import base 



class Testloader(base.Dataset):

    def __init__(
        self,
        split_file,
        return_filename=False,
        rot_index=None,
        return_type='caption',
        random_caption=False,
        caption_type='qwen',
        chunk_size=64,
        double_chunks=True,
        caption_mapping_path=None
    ):

        self.gt_files = self.get_instance_filenames_scene_vox_list(split_file)
        self.return_filename = return_filename

        self.near_surface_sdf = 0.07
        self.return_type = return_type
        self.random_caption = random_caption
        self.caption_type = caption_type
        self.chunk_size = chunk_size
        self.double_chunks = double_chunks
        self.caption_mapping_path = caption_mapping_path
        self.rot_index = rot_index

        if caption_mapping_path is None:
            CAP_MAP = '/cluster/balar/abokhovkin/data/Front3D/val_scenes_450_main_qwen_camready.json'
        else:
            CAP_MAP = caption_mapping_path

        # with open('/cluster/balar/abokhovkin/data/Front3D/val_scenes_250_main.json', 'r') as fin:
        #     self.scene_captions = json.load(fin)
        # with open('/cluster/balar/abokhovkin/data/Front3D/val_scenes_250_main_qwen.json', 'r') as fin:
        #     self.scene_captions = json.load(fin)
        with open(CAP_MAP, 'r') as fin:
            self.scene_captions = json.load(fin)
    
    def __getitem__(self, idx):

        if self.return_type == 'caption':
            output = self.get_caption(self.gt_files[idx])
        elif self.return_type == 'vox':
            output = self.get_vox(self.gt_files[idx], self.pc_size, return_sdf=self.return_sdf, idx=idx)
        if self.return_filename:
            return output, self.gt_files[idx], idx
        else:
            return output, idx

    def __len__(self):
        return len(self.point_clouds)

    def get_caption(self, f): 

        f_tokens = f.split('/')
        scene_id = f_tokens[-2]

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

        if self.random_caption:
            if self.caption_type != 'qwen':
                caption_keys = ['caption_list', 'caption_subcat_list', 'caption_list_explicit', 'caption_subcat_list_explicit', 'caption_spatial', 'caption_subcat_spatial', 'caption_room']
            else:
                caption_keys = ['caption_room_qwen', 'caption_room_cats_qwen', 'caption_rooms_subcats_qwen']
            caption_key = random.choice(caption_keys)
            caption_list = meta_data[caption_key]
        else:
            caption_key = self.scene_captions[scene_id]
            caption_list = meta_data[caption_key]

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
        
        if caption.endswith('.'):
            caption = caption[:-1]
        if caption_wall != '':
            caption = caption + ', ' + caption_wall

        return caption

    def get_vox(self, f):

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        try:
            f_tokens = f.split('/')
            f_folder = '/'.join(f_tokens[:-1])
            vox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '.npy'
            big_vox = np.load(os.path.join(f_folder, vox_filename))
            with open(f, 'r') as fin:
                meta_data = json.load(fin)
            subchunk_bounds = np.array(meta_data['subchunk_bounds'])
            gt_vox = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                             subchunk_bounds[0][1]:subchunk_bounds[1][1],
                             subchunk_bounds[0][2]:subchunk_bounds[1][2]]
        except ValueError:
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))

        if self.rot_index is None:
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
        else:
            p_rot = self.rot_index

        gt_vox = torch.FloatTensor(gt_vox)
        gt_vox = torch.rot90(gt_vox, k=p_rot, dims=[0, 2])
        gt_vox = gt_vox.numpy()
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf

        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        if gt_vox.shape != (1, chunk_size_eff, self.chunk_size, chunk_size_eff):
            gt_buf = torch.ones((1, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[:, :, :, :]
            gt_vox = gt_buf

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
