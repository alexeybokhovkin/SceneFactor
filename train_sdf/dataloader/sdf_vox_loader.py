import os
import random

import numpy as np
import json
import torch
import torch.utils.data
import torch.nn.functional as F

from . import base 



class VoxLoaderGeo(base.Dataset):

    def __init__(
        self,
        split_file,
        modulation_path=None,
        chunk_size=64,
        predefined_chunk_bounds=False,
        augment_chunks=True,
        double_chunks=True
    ):
 
        self.gt_files = self.get_instance_filenames_scene_vox_list(split_file, filter_modulation_path=modulation_path)
        self.chunk_size = chunk_size
        self.predefined_chunk_bounds = predefined_chunk_bounds
        self.augment_chunks = augment_chunks
        self.double_chunks = double_chunks

    def __getitem__(self, idx):

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        try:
            if self.predefined_chunk_bounds:
                f_tokens = self.gt_files[idx].split('/')
                f_folder = '/'.join(f_tokens[:-1])
                vox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '.npy'
                big_vox = np.load(os.path.join(f_folder, vox_filename))
                with open(self.gt_files[idx], 'r') as fin:
                    meta_data = json.load(fin)
                subchunk_bounds = np.array(meta_data['subchunk_bounds'])
                gt_vox = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                                 subchunk_bounds[0][1]:subchunk_bounds[1][1],
                                 subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            else: # preferred
                f_tokens = self.gt_files[idx].split('/')
                f_folder = '/'.join(f_tokens[:-1])
                vox_filename = f_tokens[-1].split('.')[0] + '.npy'
                big_vox = np.load(os.path.join(f_folder, vox_filename))
                h, w, d = big_vox.shape
                valid_range_min_point = [[0, 0, 0],
                                         [max(0, h - chunk_size_eff), 1, max(0, d - chunk_size_eff)]]
                random_min_point = np.random.uniform(low=valid_range_min_point[0], high=valid_range_min_point[1], size=(3,)).astype('int32')
                random_max_point = random_min_point + [chunk_size_eff, self.chunk_size, chunk_size_eff]
                gt_vox = big_vox[random_min_point[0]:random_max_point[0],
                                 random_min_point[1]:random_max_point[1],
                                 random_min_point[2]:random_max_point[2]]
        except:
            print('Value Error!')
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))

        if self.augment_chunks:
            p_sym_a = torch.rand(1)[0]
            p_sym_b = torch.rand(1)[0]
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            gt_vox = torch.FloatTensor(gt_vox)
            gt_vox = torch.rot90(gt_vox, k=p_rot, dims=[0, 2])
            gt_vox = gt_vox.numpy()
            if p_sym_a < 0.3:
                gt_vox[:, :, :] = gt_vox[::-1, :, :]
            if p_sym_b < 0.3:
                gt_vox[:, :, :] = gt_vox[:, :, ::-1]
        
        gt_vox = np.abs(gt_vox)
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox[gt_vox < -self.near_surface_sdf] = -self.near_surface_sdf
        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        num_channels = 1

        if gt_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size!')
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[0, :, :, :]
            gt_vox = gt_buf

        # for lowres training
        # gt_vox = gt_vox[:, ::4, ::4, ::4]

        data_dict = {
                    "gt_vox" : gt_vox,
                    "filename" : self.gt_files[idx],
                    "idx" : idx
                    }

        return data_dict
    
    def __len__(self):
        return len(self.gt_files)
    


class VoxLoaderSem(base.Dataset):

    def __init__(
        self,
        split_file,
        modulation_path=None,
        chunk_size=16,
        predefined_chunk_bounds=False,
        augment_chunks=True,
        double_chunks=True
    ):
 
        self.gt_files = self.get_instance_filenames_scene_vox_list(split_file, filter_modulation_path=modulation_path)
        self.chunk_size = chunk_size
        self.predefined_chunk_bounds = predefined_chunk_bounds
        self.augment_chunks = augment_chunks
        self.double_chunks = double_chunks

    def __getitem__(self, idx):

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        try:
            if self.predefined_chunk_bounds:
                semvox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '_semantic.npy'
                big_semvox = np.load(os.path.join(f_folder, semvox_filename))
                with open(self.gt_files[idx], 'r') as fin:
                    meta_data = json.load(fin)
                subchunk_bounds = np.array(meta_data['subchunk_bounds'])
                sem_vox = big_semvox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                                     subchunk_bounds[0][1]:subchunk_bounds[1][1],
                                     subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            else: # preferred
                f_tokens = self.gt_files[idx].split('/')
                f_folder = '/'.join(f_tokens[:-1])
                semvox_filename = f_tokens[-1].split('.')[0] + '_semantic.npy'
                big_semvox = np.load(os.path.join(f_folder, semvox_filename))
                h, w, d = big_semvox.shape
                valid_range_min_point = [[0, 0, 0],
                                         [max(0, h - chunk_size_eff), 1, max(0, d - chunk_size_eff)]]
                random_min_point = np.random.uniform(low=valid_range_min_point[0], high=valid_range_min_point[1], size=(3,)).astype('int32')
                random_max_point = random_min_point + [chunk_size_eff, self.chunk_size, chunk_size_eff]
                sem_vox = big_semvox[random_min_point[0]:random_max_point[0],
                                     random_min_point[1]:random_max_point[1],
                                     random_min_point[2]:random_max_point[2]]
        except ValueError:
            print('Value Error!')
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
        
        if self.augment_chunks:
            p_sym_a = torch.rand(1)[0]
            p_sym_b = torch.rand(1)[0]
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            sem_vox = torch.FloatTensor(sem_vox)
            sem_vox = torch.rot90(sem_vox, k=p_rot, dims=[0, 2])
            sem_vox = sem_vox.numpy()
            if p_sym_a < 0.3:
                sem_vox[:, :, :] = sem_vox[::-1, :, :]
            if p_sym_b < 0.3:
                sem_vox[:, :, :] = sem_vox[:, :, ::-1]

        sem_vox = torch.LongTensor(sem_vox)
        onehot_sem = F.one_hot(sem_vox, num_classes=10)
        onehot_sem = torch.permute(onehot_sem, (3, 0, 1, 2)).float()
        gt_vox = onehot_sem

        num_channels = gt_vox.shape[0]
        if gt_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            gt_buf = torch.zeros((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff))
            if num_channels > 1:
                gt_buf[:, :, :, :] = 0
                gt_buf[:, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[:, :, :, :]
                gt_vox = gt_buf

        # from training chunks
        # gt_vox = gt_vox[:, ::4, ::4, ::4]
        # from original chunks
        # gt_vox = gt_vox[:, ::8, ::8, ::8]

        data_dict = {
                    "gt_vox" : gt_vox,
                    "filename" : self.gt_files[idx],
                    "idx" : idx
                    }

        return data_dict

    def __len__(self):
        return len(self.gt_files)



class SdfVoxLoaderShapeGlot(base.Dataset):

    def __init__(
        self,
        split_file,
        modulation_path=None,
        chunk_size=64,
        augment_chunks=True,
        double_chunks=True
    ):
 
        self.gt_files = self.get_instance_filenames_scene_vox_list(split_file, filter_modulation_path=modulation_path)
        self.chunk_size = chunk_size
        self.double_chunks = double_chunks
        self.augment_chunks = augment_chunks

    def __len__(self):
        return len(self.gt_files)
    
    def __getitem__(self, idx):

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        try:
            f_tokens = self.gt_files[idx].split('/')
            f_folder = '/'.join(f_tokens[:-1])
            vox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '.npy'
            big_vox = np.load(os.path.join(f_folder, vox_filename))
            with open(self.gt_files[idx], 'r') as fin:
                meta_data = json.load(fin)
            subchunk_bounds = np.array(meta_data['subchunk_bounds'])
            gt_vox = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                             subchunk_bounds[0][1]:subchunk_bounds[1][1],
                             subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            
            idx_1 = np.random.choice(np.arange(len(self.gt_files)))
            f_tokens = self.gt_files[idx_1].split('/')
            f_folder = '/'.join(f_tokens[:-1])
            vox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '.npy'
            big_vox = np.load(os.path.join(f_folder, vox_filename))
            with open(self.gt_files[idx_1], 'r') as fin:
                meta_data_1 = json.load(fin)
            subchunk_bounds = np.array(meta_data_1['subchunk_bounds'])
            gt_vox_1 = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                               subchunk_bounds[0][1]:subchunk_bounds[1][1],
                               subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            
            idx_2 = np.random.choice(np.arange(len(self.gt_files)))
            f_tokens = self.gt_files[idx_2].split('/')
            f_folder = '/'.join(f_tokens[:-1])
            vox_filename = '_'.join(f_tokens[-1].split('_')[:-1]) + '.npy'
            big_vox = np.load(os.path.join(f_folder, vox_filename))
            with open(self.gt_files[idx_2], 'r') as fin:
                meta_data_2 = json.load(fin)
            subchunk_bounds = np.array(meta_data_2['subchunk_bounds'])
            gt_vox_2 = big_vox[subchunk_bounds[0][0]:subchunk_bounds[1][0],
                               subchunk_bounds[0][1]:subchunk_bounds[1][1],
                               subchunk_bounds[0][2]:subchunk_bounds[1][2]]
            
        except ValueError:
            print('Value Error!')
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
            gt_vox_1 = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
            gt_vox_2 = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))

        if self.self.augment_chunks:
            p_sym_a = torch.rand(1)[0]
            p_sym_b = torch.rand(1)[0]
            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            gt_vox = torch.FloatTensor(gt_vox)
            gt_vox = torch.rot90(gt_vox, k=p_rot, dims=[0, 2])
            gt_vox = gt_vox.numpy()
            if p_sym_a < 0.3:
                gt_vox[:, :, :] = gt_vox[::-1, :, :]
            if p_sym_b < 0.3:
                gt_vox[:, :, :] = gt_vox[:, :, ::-1]

            gt_vox_1 = torch.FloatTensor(gt_vox_1)
            gt_vox_1 = torch.rot90(gt_vox_1, k=p_rot, dims=[0, 2])
            gt_vox_1 = gt_vox_1.numpy()
            if p_sym_a < 0.3:
                gt_vox_1[:, :, :] = gt_vox_1[::-1, :, :]
            if p_sym_b < 0.3:
                gt_vox_1[:, :, :] = gt_vox_1[:, :, ::-1]

            gt_vox_2 = torch.FloatTensor(gt_vox_2)
            gt_vox_2 = torch.rot90(gt_vox_2, k=p_rot, dims=[0, 2])
            gt_vox_2 = gt_vox_2.numpy()
            if p_sym_a < 0.3:
                gt_vox_2[:, :, :] = gt_vox_2[::-1, :, :]
            if p_sym_b < 0.3:
                gt_vox_2[:, :, :] = gt_vox_2[:, :, ::-1]

        
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox[gt_vox < -self.near_surface_sdf] = -self.near_surface_sdf
        gt_vox = np.abs(gt_vox)
        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        gt_vox_1[gt_vox_1 > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox_1[gt_vox_1 < -self.near_surface_sdf] = -self.near_surface_sdf
        gt_vox_1 = np.abs(gt_vox_1)
        gt_vox_1 *= (1. / self.near_surface_sdf)
        gt_vox_1 = 1. - gt_vox_1
        gt_vox_1 = torch.FloatTensor(gt_vox_1)[None, ...]

        gt_vox_2[gt_vox_2 > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox_2[gt_vox_2 < -self.near_surface_sdf] = -self.near_surface_sdf
        gt_vox_2 = np.abs(gt_vox_2)
        gt_vox_2 *= (1. / self.near_surface_sdf)
        gt_vox_2 = 1. - gt_vox_2
        gt_vox_2 = torch.FloatTensor(gt_vox_2)[None, ...]

        num_channels = 1

        if gt_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size!', gt_vox.shape)
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[0, :, :, :]
            gt_vox = gt_buf

        if gt_vox_1.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size (1)!')
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox_1.shape[1], 0:gt_vox_1.shape[2], 0:gt_vox_1.shape[3]] = gt_vox_1[0, :, :, :]
            gt_vox_1 = gt_buf

        if gt_vox_2.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size (2)!')
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox_2.shape[1], 0:gt_vox_2.shape[2], 0:gt_vox_2.shape[3]] = gt_vox_2[0, :, :, :]
            gt_vox_2 = gt_buf

        true_answer = torch.LongTensor(np.array([1, 0, 0]))
        gt_voxes = torch.vstack([gt_vox, gt_vox_1, gt_vox_2])
        rand_perm = torch.randperm(3)
        true_answer = true_answer[rand_perm]
        gt_voxes = gt_voxes[rand_perm]

        try:
            caption_keys = ['caption_list', 'caption_subcat_list', 'caption_list_explicit', 'caption_subcat_list_explicit', 
                            'caption_spatial', 'caption_subcat_spatial', 'caption_room', 'caption_list_inexact', 'caption_subcat_list_inexact']
            caption_key = random.choice(caption_keys)
            caption_list = meta_data[caption_key]
            if not isinstance(caption_list, list):
                caption_list = caption_list.split(', ')
            random.shuffle(caption_list)
            caption = ''
            for item in caption_list:
                caption = caption + item + ', '
            caption = caption[:-2]
            caption_walls = meta_data['caption_layout']
        except:
            caption = ''
            caption_walls = ['', '', '', '']

        p_rot = torch.randint(low=0, high=4, size=(1,))[0]
        caption_wall = caption_walls[p_rot]
        if caption_wall != '':
            caption = caption + ', ' + caption_wall

        data_dict = {
                    "gt_voxes" : gt_voxes,
                    "true_answer" : true_answer,
                    "filename" : self.gt_files[idx],
                    "idx" : idx,
                    "caption" : caption
                    }

        return data_dict


class SdfVoxLoaderShapeGlotTest(base.Dataset):

    def __init__(
        self,
        gt_dir,
        ours_dir,
        sdfusion_dir,
        mode='sdfusion',
        chunk_size=64,
        double_chunks=True,
        caption_dir=None,
        compare_sets=('gt', 'ours')
    ):

        self.mode = mode
        self.chunk_size = chunk_size
        self.double_chunks = double_chunks
        self.caption_dir = caption_dir
        self.compare_sets = compare_sets
        self.gt_dir = gt_dir
        self.gt_files = {}
        self.gt_scenes = []
        for scene_id in os.listdir(self.gt_dir):
            if scene_id.endswith('.txt') or scene_id.endswith('.json'):
                continue
            json_files = [x for x in os.listdir(os.path.join(self.gt_dir, scene_id)) if x.endswith('.json')]
            for json_file in json_files:
                if scene_id not in self.gt_files:
                    self.gt_scenes += [scene_id]
                    self.gt_files[scene_id] = {}
                chunk_tokens = json_file.split('.')[0].split('_')
                self.gt_files[scene_id][f'{chunk_tokens[0]}_{chunk_tokens[2]}'] = os.path.join(self.gt_dir, scene_id, json_file.split('.')[0] + '.npy')

        self.ours_dir = ours_dir
        self.ours_files = {}
        self.ours_scenes = []
        for scene_id in os.listdir(self.ours_dir):
            if scene_id.endswith('.txt') or scene_id.endswith('.json'):
                continue
            npy_files = [x for x in os.listdir(os.path.join(self.ours_dir, scene_id)) if x.endswith('_full_geo.npy')]
            for npy_file in npy_files:
                if scene_id not in self.ours_files:
                    self.ours_scenes += [scene_id]
                    self.ours_files[scene_id] = {}
                chunk_tokens = npy_file.split('.')[0].split('_')
                self.ours_files[scene_id][f'{chunk_tokens[0]}_{chunk_tokens[1]}'] = os.path.join(self.ours_dir, scene_id, npy_file.split('.')[0] + '.npy')

        if self.mode == 'sdfusion':
            # sdfusion
            self.sdfusion_dir = sdfusion_dir
            self.sdfusion_files = {}
            self.sdfusion_scenes = []
            for scene_id in os.listdir(self.sdfusion_dir):
                if scene_id.endswith('.txt') or scene_id.endswith('.json'):
                    continue
                npy_files = [x for x in os.listdir(os.path.join(self.sdfusion_dir, scene_id)) if x.endswith('_geo.npy') and 'full_geo' not in x]
                for npy_file in npy_files:
                    if scene_id not in self.sdfusion_files:
                        self.sdfusion_scenes += [scene_id]
                        self.sdfusion_files[scene_id] = {}
                    chunk_tokens = npy_file.split('.')[0].split('_')
                    self.sdfusion_files[scene_id][f'{chunk_tokens[0]}_{chunk_tokens[1]}'] = os.path.join(self.sdfusion_dir, scene_id, npy_file.split('.')[0] + '.npy')

        elif self.mode == 'text2room/atiss':
            # text2room, atiss
            self.sdfusion_dir = sdfusion_dir
            self.sdfusion_files = {}
            self.sdfusion_scenes = []
            for scene_id in os.listdir(self.sdfusion_dir):
                if scene_id.endswith('.txt') or scene_id.endswith('.json') or scene_id == 'output':
                    continue
                npy_files = [x for x in os.listdir(os.path.join(self.sdfusion_dir, scene_id)) if x.endswith('.npy')]
                for npy_file in npy_files:
                    if scene_id not in self.sdfusion_files:
                        self.sdfusion_scenes += [scene_id]
                        self.sdfusion_files[scene_id] = {}
                    chunk_tokens = npy_file.split('.')[0].split('_')
                    self.sdfusion_files[scene_id][f'{chunk_tokens[1]}_{chunk_tokens[3]}'] = os.path.join(self.sdfusion_dir, scene_id, npy_file.split('.')[0] + '.npy')

        self.intersect_scenes = [x for x in self.ours_scenes if x in self.gt_scenes and x in self.sdfusion_scenes]
        gt_keys = list(self.gt_files.keys())
        for scene_id in gt_keys:
            if scene_id not in self.intersect_scenes:
                del self.gt_files[scene_id]
        ours_keys = list(self.ours_files.keys())
        for scene_id in ours_keys:
            if scene_id not in self.intersect_scenes:
                del self.ours_files[scene_id]
        sdfusion_keys = list(self.sdfusion_files.keys())
        for scene_id in sdfusion_keys:
            if scene_id not in self.intersect_scenes:
                print(scene_id)
                del self.sdfusion_files[scene_id]
        for scene_id in self.intersect_scenes:
            gt_keys = list(self.gt_files[scene_id].keys())
            for chunk_name in gt_keys:
                if chunk_name not in self.ours_files[scene_id] or chunk_name not in self.sdfusion_files[scene_id]:
                    del self.gt_files[scene_id][chunk_name]
        for scene_id in self.intersect_scenes:
            ours_keys = list(self.ours_files[scene_id].keys())
            for chunk_name in ours_keys:
                if chunk_name not in self.gt_files[scene_id]:
                    del self.ours_files[scene_id][chunk_name]
        for scene_id in self.intersect_scenes:
            sdfusion_keys = list(self.sdfusion_files[scene_id].keys())
            for chunk_name in sdfusion_keys:
                if chunk_name not in self.gt_files[scene_id]:
                    del self.sdfusion_files[scene_id][chunk_name]

        self.gt_files_flat = []
        self.ours_files_flat = []
        self.sdfusion_files_flat = []
        for scene_id in self.gt_files:
            for chunk_name in self.gt_files[scene_id]:
                self.gt_files_flat += [self.gt_files[scene_id][chunk_name]]
                self.ours_files_flat += [self.ours_files[scene_id][chunk_name]]
                self.sdfusion_files_flat += [self.sdfusion_files[scene_id][chunk_name]]

        if caption_dir is None:
            self.caption_dir = '/cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/3dfuture_1ch_lowlowres_newdec_onlysem_nll_vqorig_2x2_allcap_l1_final_2/recon_semantic_3steps'
        else:
            self.caption_dir = caption_dir

        print('gt_files', len(self.gt_files_flat))
        print('ours_files', len(self.ours_files_flat))
        print('sdfusion_files', len(self.sdfusion_files_flat))

    def __len__(self):
        return len(self.gt_files_flat)

    def __getitem__(self, idx):

        if self.double_chunks:
            chunk_size_eff = 2 * self.chunk_size
        else:
            chunk_size_eff = self.chunk_size

        try:
            f_tokens = self.gt_files_flat[idx].split('/') # gt file
            vox_filename = self.gt_files_flat[idx]
            gt_vox = np.load(vox_filename)[:, :self.chunk_size, :]

            scene_id = f_tokens[-2]
            chunk_id = '_'.join(f_tokens[-1].split('_')[:2])
            
            f_tokens = self.ours_files_flat[idx].split('/') # ours files
            vox_filename = self.ours_files_flat[idx]
            ours_vox = np.load(vox_filename)

            f_tokens = self.sdfusion_files_flat[idx].split('/') # sdfusion files
            vox_filename = self.sdfusion_files_flat[idx]
            sdfusion_vox = np.squeeze(np.load(vox_filename))[:, :self.chunk_size, :]

            try:
                meta_data_file = os.path.join(self.caption_dir, scene_id, f'{chunk_id}_recon.json')
                with open(meta_data_file, 'r') as fin:
                    caption = json.load(fin)['caption']
            except FileNotFoundError:
                print('Not found!')
                caption = 'Empty room'
            
        except ValueError:
            print('Value Error!')
            gt_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
            ours_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))
            sdfusion_vox = np.ones((chunk_size_eff, self.chunk_size, chunk_size_eff))

        
        gt_vox[gt_vox > self.near_surface_sdf] = self.near_surface_sdf
        gt_vox[gt_vox < -self.near_surface_sdf] = -self.near_surface_sdf
        gt_vox = np.abs(gt_vox)
        gt_vox *= (1. / self.near_surface_sdf)
        gt_vox = 1. - gt_vox
        gt_vox = torch.FloatTensor(gt_vox)[None, ...]

        ours_vox = 1. - ours_vox
        ours_vox = ours_vox / 0.18 * 1 / 80
        ours_vox[ours_vox > self.near_surface_sdf] = self.near_surface_sdf
        ours_vox[ours_vox < -self.near_surface_sdf] = -self.near_surface_sdf
        ours_vox = np.abs(ours_vox)
        ours_vox *= (1. / self.near_surface_sdf)
        ours_vox = 1. - ours_vox
        ours_vox = torch.FloatTensor(ours_vox)[None, ...]

        sdfusion_vox[sdfusion_vox > self.near_surface_sdf] = self.near_surface_sdf
        sdfusion_vox[sdfusion_vox < -self.near_surface_sdf] = -self.near_surface_sdf
        sdfusion_vox = np.abs(sdfusion_vox)
        sdfusion_vox *= (1. / self.near_surface_sdf)
        sdfusion_vox = 1. - sdfusion_vox
        sdfusion_vox = torch.FloatTensor(sdfusion_vox)[None, ...]

        num_channels = 1

        if gt_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size!', gt_vox.shape)
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:gt_vox.shape[1], 0:gt_vox.shape[2], 0:gt_vox.shape[3]] = gt_vox[0, :, :, :]
            gt_vox = gt_buf

        if ours_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size (1)!')
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:ours_vox.shape[1], 0:ours_vox.shape[2], 0:ours_vox.shape[3]] = ours_vox[0, :, :, :]
            ours_vox = gt_buf

        if sdfusion_vox.shape != (num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff):
            print('Incorrect size (2)!')
            gt_buf = torch.ones((num_channels, chunk_size_eff, self.chunk_size, chunk_size_eff)) * self.near_surface_sdf
            gt_buf[0, 0:sdfusion_vox.shape[1], 0:sdfusion_vox.shape[2], 0:sdfusion_vox.shape[3]] = sdfusion_vox[0, :, :, :]
            sdfusion_vox = gt_buf


        true_answer = torch.LongTensor(np.array([1, 0, 0]))
        if 'gt' in self.compare_sets and 'ours' in self.compare_sets:
            gt_voxes = torch.vstack([gt_vox, ours_vox])
        elif 'sdfusion' in self.compare_sets and 'ours' in self.compare_sets:
            gt_voxes = torch.vstack([ours_vox, sdfusion_vox])
        elif 'sdfusion' in self.compare_sets and 'gt' in self.compare_sets:
            gt_voxes = torch.vstack([gt_vox, sdfusion_vox])

        data_dict = {
                    "gt_voxes" : gt_voxes,
                    "true_answer" : true_answer,
                    "filename" : self.gt_files_flat[idx],
                    "idx" : idx,
                    "caption" : caption
                    }

        return data_dict
