import os
import numpy as np
import trimesh
import skimage
import json
from tqdm import tqdm


SAVE_DIR = '/cluster/balar/abokhovkin/data/Front3D'
GT_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference'
scene_ids = [x for x in os.listdir(GT_DIR) if not x.endswith('.txt')]

all_points = []
all_points_filenames = []
i = 0
for scene_id in tqdm(scene_ids):
    if i > 2048:
        break
    npy_files = [x for x in os.listdir(os.path.join(GT_DIR, scene_id)) if x.endswith('.json')]
    npy_files = [x.split('.')[0] + '.npy' for x in npy_files]
    
    for npy_file in npy_files:
        sdf = np.load(os.path.join(GT_DIR, scene_id, npy_file))
        
        try:
            level = 1. / 80
            vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)
            mesh_new = trimesh.Trimesh(vertices, faces)
        except ValueError:
            continue
        
        sampled_points, _ = trimesh.sample.sample_surface(mesh_new, 4096)
        points_mean = sampled_points.mean(axis=0)
        point_min = sampled_points.min(axis=0)
        sampled_points = sampled_points - point_min
        points_max = sampled_points[:, 1].max()
        sampled_points = sampled_points / points_max
        all_points += [sampled_points.astype('float16').tolist()]
        all_points_filenames += [(scene_id, npy_file.split('.')[0])]

        i += 1

with open(os.path.join(SAVE_DIR, 'gt_pc_16k.json'), 'w') as fout:
    json.dump({'points': all_points}, fout)
with open(os.path.join(SAVE_DIR, 'gt_pc_16k_filenames.txt'), 'w') as fout:
    for filename in all_points_filenames:
        fout.write(f"{filename}\n")