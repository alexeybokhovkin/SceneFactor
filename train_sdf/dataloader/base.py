import numpy as np
import logging
import os
import torch
import torch.utils.data
from torch.nn import functional as F

import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_file, # json filepath which contains train/test classes and meshes 
        subsample,
        gt_filename,
        #pc_size=1024,
    ):

        self.data_source = data_source 
        self.subsample = subsample
        self.split_file = split_file
        self.gt_filename = gt_filename
        #self.pc_size = pc_size

        # example
        # data_source: "data"
        # ws.sdf_samples_subdir: "SdfSamples"
        # self.gt_files[0]: "acronym/couch/meshname/sdf_data.csv"
            # with gt_filename="sdf_data.csv"

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):     
        return NotImplementedError

    def sample_pointcloud(self, csvfile, pc_size):
        f = pd.read_csv(csvfile, sep=',',header=None).values[1:]

        f = f[np.abs(f[:,-1]) <= 0.005][:,:3]

        if f.shape[0] < pc_size:
            pc_idx = np.random.choice(f.shape[0], pc_size)
        else:
            pc_idx = np.random.choice(f.shape[0], pc_size, replace=False)

        return torch.from_numpy(f[pc_idx]).float()

    def labeled_sampling(self, f, subsample, pc_size=1024, load_from_path=True, return_sdf=False):
        if load_from_path:
            f = pd.read_csv(f, sep=',',header=None).values[1:]
            f = torch.from_numpy(f)

        half = int(subsample / 2) 
        neg_tensor = f[f[:,-1]<0]
        pos_tensor = f[f[:,-1]>0]

        if len(pos_tensor) == 0:
            pos_tensor = torch.FloatTensor([[0, 0, 0, 0.7],
                                            [0, 0, 0, 0.7]])
        if len(neg_tensor) == 0:
            neg_tensor = torch.FloatTensor([[0, 0, 0, 0.7],
                                            [0, 0, 0, 0.7]])

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]

        if neg_tensor.shape[0] == 0:
            neg_sample = pos_tensor[neg_idx]
        else:
            neg_sample = neg_tensor[neg_idx]

        pc = f[np.abs(f[:,-1]) <= 0.005][:,:3]
        if len(pc) < pc_size:
            pc_idx = np.argsort(np.abs(f[:, 3]))[:pc_size]
            pc = f[pc_idx][:, :3]
        else:
            pc_idx = torch.randperm(pc.shape[0])[:pc_size]
            pc = pc[pc_idx]

        if len(pc) == 0:
            pc = torch.FloatTensor([[0, 0, 0],
                                    [0, 0, 0]])

        if len(pc) < pc_size:
            indices = np.random.choice(len(pc), pc_size)
            pc = pc[indices]

        samples = torch.cat([pos_sample, neg_sample], 0)
        if len(samples) < subsample:
            all_idx = torch.randint(0, samples.shape[0], (subsample,))
            samples = samples[all_idx]

        near_surf_samples = f[np.abs(f[:, -1]) <= 0.1]
        if len(near_surf_samples) == 0:
            near_surf_samples = torch.FloatTensor([[0, 0, 0, 0.7],
                                                   [0, 0, 0, 0.7]])
        if len(near_surf_samples) < pc_size:
            near_surf_idx = np.argsort(np.abs(f[:, 3]))[:pc_size]
            near_surf_samples = f[near_surf_idx]
        else:
            near_surf_idx = torch.randperm(near_surf_samples.shape[0])[:pc_size]
            near_surf_samples = near_surf_samples[near_surf_idx]
        if len(near_surf_samples) < pc_size:
            print('near_surf_samples (reused)', near_surf_samples.shape)
            print('f (reused)', f.shape, f[:, 3].min(), f[:, 3].max())
            near_surf_idx = torch.randint(0, near_surf_samples.shape[0], (pc_size,))
            near_surf_samples = near_surf_samples[near_surf_idx]


        if return_sdf:
            pc = near_surf_samples

        if len(pc) < pc_size:
            print('pc', pc.shape)
        if len(near_surf_samples) < pc_size:
            print('near_surf_samples', near_surf_samples.shape)
            print('f', f.shape, f[:, 3].min(), f[:, 3].max())
        if len(samples) < subsample:
            print('samples', samples.shape)

        pc_output = pc.float().squeeze()
        xyz_output = samples[:,:3].float().squeeze()
        sdv_output = samples[:, 3].float().squeeze()

        # make unsigned distance field
        sdv_output = torch.abs(sdv_output)

        return pc_output, xyz_output, sdv_output # pc, xyz, sdv
    

    def labeled_sampling_sdf(self, f, subsample, pc_size=1024, load_from_path=True, return_sdf=False, sdf_trunc=None):

        xx = np.array(f['x'])
        yy = np.array(f['y'])
        zz = np.array(f['z'])
        sdf = np.array(f['sdf'])
        xyzd = np.vstack([xx, yy, zz, sdf]).T
        xyzd = torch.FloatTensor(xyzd)

        f = xyzd[np.abs(xyzd[:, -1]) <= sdf_trunc]

        half = int(subsample / 2) 
        neg_tensor = f[f[:, -1] < 0]
        pos_tensor = f[f[:, -1] > 0]

        if len(pos_tensor) == 0:
            pos_tensor = torch.FloatTensor([[0, 0, 0, 0.06],
                                            [0, 0, 0, 0.06]])
        if len(neg_tensor) == 0:
            neg_tensor = torch.FloatTensor([[0, 0, 0, 0.06],
                                            [0, 0, 0, 0.06]])

        if pos_tensor.shape[0] < half:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half]

        if neg_tensor.shape[0] < half:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half]

        pos_sample = pos_tensor[pos_idx]

        if neg_tensor.shape[0] == 0:
            neg_sample = pos_tensor[neg_idx]
        else:
            neg_sample = neg_tensor[neg_idx]

        pc = f[np.abs(f[:,-1]) <= 0.01][:,:3]
        if len(pc) < pc_size:
            pc_idx = np.argsort(np.abs(f[:, 3]))[:pc_size]
            pc = f[pc_idx][:, :3]
        else:
            pc_idx = torch.randperm(pc.shape[0])[:pc_size]
            pc = pc[pc_idx]

        if len(pc) == 0:
            pc = torch.FloatTensor([[0, 0, 0],
                                    [0, 0, 0]])

        if len(pc) < pc_size:
            indices = np.random.choice(len(pc), pc_size)
            pc = pc[indices]

        samples = torch.cat([pos_sample, neg_sample], 0)
        if len(samples) < subsample:
            all_idx = torch.randint(0, samples.shape[0], (subsample,))
            samples = samples[all_idx]

        near_surf_samples = f[np.abs(f[:, -1]) <= 0.01]
        if len(near_surf_samples) == 0:
            near_surf_samples = torch.FloatTensor([[0, 0, 0, 0.06],
                                                   [0, 0, 0, 0.06]])
        if len(near_surf_samples) < pc_size:
            near_surf_idx = np.argsort(np.abs(f[:, 3]))[:pc_size]
            near_surf_samples = f[near_surf_idx]
        else:
            near_surf_idx = torch.randperm(near_surf_samples.shape[0])[:pc_size]
            near_surf_samples = near_surf_samples[near_surf_idx]
        if len(near_surf_samples) < pc_size:
            print('near_surf_samples (reused)', near_surf_samples.shape)
            print('f (reused)', f.shape, f[:, 3].min(), f[:, 3].max())
            near_surf_idx = torch.randint(0, near_surf_samples.shape[0], (pc_size,))
            near_surf_samples = near_surf_samples[near_surf_idx]


        if return_sdf:
            pc = near_surf_samples
            pc[:, 3] *= 100

        if len(pc) < pc_size:
            print('pc 2', pc.shape)
        if len(near_surf_samples) < pc_size:
            print('near_surf_samples', near_surf_samples.shape)
            print('f', f.shape, f[:, 3].min(), f[:, 3].max())
        if len(samples) < subsample:
            print('samples', samples.shape)

        pc_output = pc.float().squeeze()
        xyz_output = samples[:,:3].float().squeeze()
        sdv_output = samples[:, 3].float().squeeze()

        # make unsigned distance field
        # sdv_output = torch.abs(sdv_output)

        return pc_output, xyz_output, sdv_output # pc, xyz, sdv
    

    def labeled_sampling_vox(self, gt_vox, num_surface_points, num_grid_points, pc_size, return_sdf=True, truncation_thr=0.06, num_sampling_points=150000):

        half_points = num_surface_points // 2
        gt_vox = torch.FloatTensor(gt_vox)
        gt_vox = torch.clamp(gt_vox, min=-truncation_thr-0.02, max=truncation_thr+0.02)
        sampled_points = torch.rand((1, num_sampling_points, 1, 1, 3)) * 2 - 1.0
        sampled_xyz = sampled_points[0, :, 0, 0, :]

        sampled_sdf = F.grid_sample(gt_vox[None, None, ...], sampled_points, padding_mode='border', align_corners=True, mode='bilinear')
        sampled_sdf = sampled_sdf.squeeze()

        sampled_surface_points = torch.where(torch.abs(sampled_sdf) < truncation_thr)
        num_sampled_surface_points = len(sampled_surface_points)

        near_surf_samples = torch.cat([sampled_xyz[sampled_surface_points], sampled_sdf[sampled_surface_points][:, None]], dim=1)
        all_samples = torch.cat([sampled_xyz, sampled_sdf[:, None]], dim=1)

        grid_idx = torch.randperm(all_samples.shape[0])[:num_grid_points]
        grid_samples = all_samples[grid_idx]

        if num_sampled_surface_points < num_surface_points:
            sampled_points_2 = torch.rand((1, num_sampling_points, 1, 1, 3)) * 2 - 1.0
            sampled_xyz_2 = sampled_points_2[0, :, 0, 0, :]
            sampled_sdf_2 = F.grid_sample(gt_vox[None, None, ...], sampled_points_2, padding_mode='border', align_corners=True, mode='bilinear')
            sampled_sdf_2 = sampled_sdf_2.squeeze()

            sampled_surface_points_2 = torch.where(torch.abs(sampled_sdf_2) < truncation_thr)
            num_sampled_surface_points_2 = len(sampled_surface_points_2)
            num_sampled_surface_points += num_sampled_surface_points_2
            sampled_xyz = torch.cat([sampled_xyz, sampled_xyz_2], dim=0)

            near_surf_samples_2 = torch.cat([sampled_xyz_2[sampled_surface_points_2], sampled_sdf_2[sampled_surface_points_2][:, None]], dim=1)
            near_surf_samples = torch.cat([near_surf_samples, near_surf_samples_2], dim=0)

        if len(near_surf_samples) < 2:
            near_surf_samples = torch.FloatTensor([[0, 0, 0, truncation_thr],
                                                   [0, 0, 0, truncation_thr]])

        if num_sampled_surface_points < num_surface_points:
            near_surf_idx = torch.randint(0, near_surf_samples.shape[0], (num_surface_points,))
            near_surf_samples = near_surf_samples[near_surf_idx]

        pc = near_surf_samples[np.abs(near_surf_samples[:, -1]) <= 0.01]
        if len(pc) < pc_size:
            pc_idx = np.argsort(np.abs(near_surf_samples[:, 3]))[:pc_size]
            pc = near_surf_samples[pc_idx]
        else:
            pc_idx = torch.randperm(pc.shape[0])[:pc_size]
            pc = pc[pc_idx]

        if len(pc) == 0:
            pc = torch.FloatTensor([[0, 0, 0, truncation_thr],
                                    [0, 0, 0, truncation_thr]])
            
        neg_tensor = near_surf_samples[near_surf_samples[:, -1] < 0]
        pos_tensor = near_surf_samples[near_surf_samples[:, -1] > 0]

        # print('points', pos_tensor.shape, neg_tensor.shape, grid_samples.shape)

        if pos_tensor.shape[0] < half_points:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half_points,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half_points]

        if neg_tensor.shape[0] < half_points:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half_points] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half_points,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half_points]

        pos_sample = pos_tensor[pos_idx]

        if neg_tensor.shape[0] == 0:
            neg_sample = pos_tensor[neg_idx]
        else:
            neg_sample = neg_tensor[neg_idx]

        surface_samples = torch.cat([pos_sample, neg_sample], 0)

        if return_sdf:
            pc[:, 3] *= 1
        else:
            pc = pc[:, :3]

        # make unsigned distance field
        surface_samples[:, 3] = torch.abs(surface_samples[:, 3])
        grid_samples[:, 3] = torch.abs(grid_samples[:, 3])
        pc[:, 3] = torch.abs(pc[:, 3])

        # make unsigned distance field
        # surface_samples[:, 3][surface_samples[:, 3] < 0] = 0.
        # grid_samples[:, 3][grid_samples[:, 3] < 0] = 0.
        # pc[:, 3][pc[:, 3] < 0] = 0.

        return surface_samples, grid_samples, pc
    
    def labeled_sampling_sem_vox(self, gt_vox, sem_vox, num_surface_points, num_grid_points, pc_size, return_sdf=True, truncation_thr=0.06, num_sampling_points=150000):

        half_points = num_surface_points // 2
        gt_vox = torch.FloatTensor(gt_vox)
        gt_vox = torch.clamp(gt_vox, min=-truncation_thr-0.02, max=truncation_thr+0.02)
        sampled_points = torch.rand((1, num_sampling_points, 1, 1, 3)) * 2 - 1.0
        sampled_xyz = sampled_points[0, :, 0, 0, :]

        sampled_sdf = F.grid_sample(gt_vox[None, ...], sampled_points, padding_mode='border', align_corners=True, mode='bilinear')
        sampled_sdf = sampled_sdf.squeeze()
        all_sampled_sem = F.grid_sample(sem_vox[None, ...], sampled_points, padding_mode='border', align_corners=True, mode='nearest')
        all_sampled_sem = all_sampled_sem.squeeze()
        all_sampled_sem = torch.permute(all_sampled_sem, (1, 0))

        sampled_surface_points = torch.where(torch.abs(sampled_sdf) < truncation_thr)
        num_sampled_surface_points = len(sampled_surface_points)

        near_surf_samples = torch.cat([sampled_xyz[sampled_surface_points], sampled_sdf[sampled_surface_points][:, None]], dim=1)
        all_samples = torch.cat([sampled_xyz, sampled_sdf[:, None]], dim=1)
        sampled_sem = all_sampled_sem[sampled_surface_points]

        grid_idx = torch.randperm(all_samples.shape[0])[:num_grid_points]
        grid_samples = all_samples[grid_idx]
        grid_sampled_sem = all_sampled_sem[grid_idx]

        if num_sampled_surface_points < num_surface_points:
            sampled_points_2 = torch.rand((1, num_sampling_points, 1, 1, 3)) * 2 - 1.0
            sampled_xyz_2 = sampled_points_2[0, :, 0, 0, :]
            sampled_sdf_2 = F.grid_sample(gt_vox[None, ...], sampled_points_2, padding_mode='border', align_corners=True, mode='bilinear')
            sampled_sdf_2 = sampled_sdf_2.squeeze()
            sampled_sem_2 = F.grid_sample(sem_vox[None, ...], sampled_points_2, padding_mode='border', align_corners=True, mode='nearest')
            sampled_sem_2 = sampled_sem_2.squeeze()
            sampled_sem_2 = torch.permute(sampled_sem_2, (1, 0))

            sampled_surface_points_2 = torch.where(torch.abs(sampled_sdf_2) < truncation_thr)
            num_sampled_surface_points_2 = len(sampled_surface_points_2)
            num_sampled_surface_points += num_sampled_surface_points_2
            sampled_xyz = torch.cat([sampled_xyz, sampled_xyz_2], dim=0)

            near_surf_samples_2 = torch.cat([sampled_xyz_2[sampled_surface_points_2], sampled_sdf_2[sampled_surface_points_2][:, None]], dim=1)
            sampled_sem_2 = sampled_sem_2[sampled_surface_points_2]
            near_surf_samples = torch.cat([near_surf_samples, near_surf_samples_2], dim=0)
            sampled_sem = torch.cat([sampled_sem, sampled_sem_2], dim=0)

        if len(near_surf_samples) < 2:
            near_surf_samples = torch.FloatTensor([[0, 0, 0, truncation_thr],
                                                   [0, 0, 0, truncation_thr]])
            sampled_sem = torch.zeros((2, 3, 10)).long()

        if num_sampled_surface_points < num_surface_points:
            near_surf_idx = torch.randint(0, near_surf_samples.shape[0], (num_surface_points,))
            near_surf_samples = near_surf_samples[near_surf_idx]
            sampled_sem = sampled_sem[near_surf_idx]

        pc = near_surf_samples[np.abs(near_surf_samples[:, -1]) <= 0.01]
        if len(pc) < pc_size:
            pc_idx = np.argsort(np.abs(near_surf_samples[:, 3]))[:pc_size]
            pc = near_surf_samples[pc_idx]
        else:
            pc_idx = torch.randperm(pc.shape[0])[:pc_size]
            pc = pc[pc_idx]

        if len(pc) == 0:
            pc = torch.FloatTensor([[0, 0, 0, truncation_thr],
                                    [0, 0, 0, truncation_thr]])

        neg_tensor = near_surf_samples[near_surf_samples[:, -1] < 0]
        pos_tensor = near_surf_samples[near_surf_samples[:, -1] > 0]
        # neg_sampled_sem = sampled_sem[near_surf_samples[:, -1] < 0]
        # pos_sampled_sem = sampled_sem[near_surf_samples[:, -1] > 0]
        neg_tensor_sem = sampled_sem[near_surf_samples[:, -1] < 0]
        pos_tensor_sem = sampled_sem[near_surf_samples[:, -1] > 0]

        if pos_tensor.shape[0] < half_points:
            pos_idx = torch.randint(0, pos_tensor.shape[0], (half_points,))
        else:
            pos_idx = torch.randperm(pos_tensor.shape[0])[:half_points]

        if neg_tensor.shape[0] < half_points:
            if neg_tensor.shape[0]==0:
                neg_idx = torch.randperm(pos_tensor.shape[0])[:half_points] # no neg indices, then just fill with positive samples
            else:
                neg_idx = torch.randint(0, neg_tensor.shape[0], (half_points,))
        else:
            neg_idx = torch.randperm(neg_tensor.shape[0])[:half_points]

        try:
            pos_sample = pos_tensor[pos_idx]
            pos_sampled_sem = pos_tensor_sem[pos_idx]
        except:
            print('pos 1', pos_tensor.shape, pos_tensor_sem.shape)
            print('neg 1', neg_tensor.shape, neg_tensor_sem.shape)

        try:
            if neg_tensor.shape[0] == 0:
                neg_sample = pos_tensor[neg_idx]
                neg_sampled_sem = pos_tensor_sem[neg_idx]
            else:
                neg_sample = neg_tensor[neg_idx]
                neg_sampled_sem = neg_tensor_sem[neg_idx]
        except:
            print('pos 2', pos_tensor.shape, pos_sample.shape, pos_sampled_sem.shape)
            print('neg 2', neg_tensor.shape, neg_sample.shape, neg_sampled_sem.shape)

        surface_samples = torch.cat([pos_sample, neg_sample], 0)
        surface_sem_samples = torch.cat([pos_sampled_sem, neg_sampled_sem], 0)

        if return_sdf:
            pc[:, 3] *= 1
        else:
            pc = pc[:, :3]

        # make unsigned distance field
        surface_samples[:, 3] = torch.abs(surface_samples[:, 3])
        grid_samples[:, 3] = torch.abs(grid_samples[:, 3])
        pc[:, 3] = torch.abs(pc[:, 3])
        
        # make unsigned distance field
        # surface_samples[:, 3][surface_samples[:, 3] < 0] = 0.
        # grid_samples[:, 3][grid_samples[:, 3] < 0] = 0.
        # pc[:, 3][pc[:, 3] < 0] = 0.

        return surface_samples, grid_samples, surface_sem_samples, grid_sampled_sem, pc

    def get_instance_filenames(self, data_source, split, gt_filename="sdf_data.csv", filter_modulation_path=None):
            
            do_filter = filter_modulation_path is not None 
            csvfiles = []
            for dataset in split: # e.g. "acronym" "shapenet"
                for class_name in split[dataset]:
                    for instance_name in split[dataset][class_name]:
                        instance_filename = os.path.join(data_source, dataset, class_name, instance_name, gt_filename)

                        if do_filter:
                            # mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.txt")
                            mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.npy")


                            # do not load if the modulation does not exist; i.e. was not trained by diffusion model
                            if not os.path.isfile(mod_file):
                                continue
                        
                        if not os.path.isfile(instance_filename):
                            logging.warning("Requested non-existent file '{}'".format(instance_filename))
                            continue

                        csvfiles.append(instance_filename)
            return csvfiles
    

    def get_instance_filenames_scene(self, data_source, split, gt_filename="sdf_data.csv", filter_modulation_path=None):
            
            do_filter = filter_modulation_path is not None 
            csvfiles = []
            for dataset in split: # e.g. "acronym" "shapenet"
                for class_name in split[dataset]:
                    for scene_id in split[dataset][class_name]:
                        for instance_name in split[dataset][class_name][scene_id]:
                            instance_filename = os.path.join(data_source, dataset, class_name, scene_id, instance_name, gt_filename)

                            # filter chunks over 0th level
                            instance_name_tokens = instance_name.split('_')
                            if instance_name_tokens[1] != '0':
                                continue

                            if do_filter:
                                # mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.txt")
                                mod_file = os.path.join(filter_modulation_path, class_name, instance_name, "latent.npy")


                                # do not load if the modulation does not exist; i.e. was not trained by diffusion model
                                if not os.path.isfile(mod_file):
                                    continue
                            
                            if not os.path.isfile(instance_filename):
                                logging.warning("Requested non-existent file '{}'".format(instance_filename))
                                continue

                            csvfiles.append(instance_filename)
            return csvfiles
    

    def get_instance_filenames_scene_list(self, split, gt_filename="sdf_data.csv", trunc_filename="sdf_trunc_data.csv", filter_modulation_path=None):

        do_filter = filter_modulation_path is not None
        with open(split, 'r') as fin:
            lines = fin.readlines()
            csvfiles = [os.path.join(x[:-1], gt_filename) for x in lines]
            csvtruncfiles = [os.path.join(x[:-1], trunc_filename) for x in lines]

        return csvfiles, csvtruncfiles
    

    def get_instance_filenames_scene_vox_list(self, split, filter_modulation_path=None):

        do_filter = filter_modulation_path is not None
        with open(split, 'r') as fin:
            lines = fin.readlines()
            npyfiles = [x[:-1] for x in lines]

        return npyfiles

        
