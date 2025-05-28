import os
import numpy as np
import trimesh
import skimage
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F


def get_sem_labels(sem_vox):
    palette = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [255, 0, 0],
        4: [255, 0, 255],
        5: [255, 255, 0],
        6: [0, 255, 255],
        7: [128, 128, 255],
        8: [143, 63, 214],
        9: [255, 115, 0]
    }
    palettex16 = {key: v[0] * 256 ** 2 + v[1] * 256 + v[2] for key, v in palette.items()}
    
    sem_ids = sem_vox[sem_vox >= 1]
    coords = np.where(sem_vox >= 1)
    coords = np.vstack(coords).T / 16
    colors = [palettex16[x] for x in sem_ids]
    return coords, colors, sem_ids


SAVE_DIR = '/cluster/balar/abokhovkin/data/Front3D'
GT_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference'
scene_ids = sorted([x for x in os.listdir(GT_DIR) if not x.endswith('.txt')])

all_points = []
i = 0
for scene_id in tqdm(scene_ids):
    if i > 1024:
        break
    npy_files = [x for x in os.listdir(os.path.join(GT_DIR, scene_id)) if x.endswith('.json')]
    npy_files = [x.split('.')[0] + '_semantic.npy' for x in npy_files]
    
    for npy_file in npy_files:
        sem = np.load(os.path.join(GT_DIR, scene_id, npy_file))
        
        sem = sem[::4, ::4, ::4][:, :16, :]
        coords, colors, sem_labels = get_sem_labels(sem)
        sem_labels_tensor = torch.LongTensor(sem_labels)
        onehot_sem = F.one_hot(sem_labels_tensor, num_classes=10)

        if len(coords) < 1024:
            continue

        sample_ids = np.random.choice(np.arange(len(coords)), 1024, replace=False)
        samples_coords = coords[sample_ids]
        samples_sem_labels = onehot_sem[sample_ids].numpy().astype('float16')
        concat_samples = np.hstack([samples_coords, samples_sem_labels])
        all_points += [concat_samples.tolist()]

        i += 1

with open(os.path.join(SAVE_DIR, 'gt_pc_sem.json'), 'w') as fout:
    json.dump({'points': all_points}, fout)