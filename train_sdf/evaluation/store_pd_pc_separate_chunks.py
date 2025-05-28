import os
import numpy as np
import trimesh
import skimage
import json
from tqdm import tqdm

# This is the main file to store SceneFactor points from separate chunks

GT_DIR = '/cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/3dfuture_1ch_lowlowres_newdec_onlysem_nll_vqorig_2x2_allcap_l1_final_2/recon_separate_sem'

npy_files = [x for x in os.listdir(GT_DIR) if '_geo.npy' in x] # main results

all_points = []
all_points_filenames = []
i = 0

VIS_CHUNKS = '/cluster/andram/abokhovkin/data/Front3D/visuals/ours_separate_chunks_420000_vis'
vis_chunk_names = []
for scene_id in os.listdir(VIS_CHUNKS):
    for filename in os.listdir(os.path.join(VIS_CHUNKS, scene_id)):
        tokens = filename.split('_')
        scene_id = tokens[0]
        chunk_id = '_'.join(tokens[1:4])
        vis_chunk_names += [f"{scene_id}_{chunk_id}"]

for npy_file in tqdm(npy_files):

    if i > 2048:
        break
    sdf = np.load(os.path.join(GT_DIR, npy_file))
    sdf = np.squeeze(sdf)
    
    try:
        sdf = 1.0 - sdf
        level = 0.18
        vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)
        mesh_new = trimesh.Trimesh(vertices, faces)
    except ValueError:
        print('no level')
        continue

    tokens = npy_file.split('_')
    scene_id = tokens[0]
    chunk_id = '_'.join(tokens[1:4])

    if f"{scene_id}_{chunk_id}" not in vis_chunk_names:
        continue
    
    sampled_points, _ = trimesh.sample.sample_surface(mesh_new, 4096)
    points_mean = sampled_points.mean(axis=0)
    point_min = sampled_points.min(axis=0)
    sampled_points = sampled_points - point_min
    points_max = sampled_points[:, 1].max()
    sampled_points = sampled_points / points_max
    all_points += [sampled_points.tolist()]
    all_points_filenames += [(scene_id, chunk_id)]

    i += 1

with open(os.path.join(GT_DIR, 'pd_pc_nn_analysis.json'), 'w') as fout:
    json.dump({'points': all_points}, fout)
with open(os.path.join(GT_DIR, 'pd_pc_nn_analysis_filenames.txt'), 'w') as fout:
    for filename in all_points_filenames:
        fout.write(f"{filename}\n")