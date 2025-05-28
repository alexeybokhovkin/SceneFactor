import os, sys
import numpy as np
import trimesh
from tqdm import tqdm
import json
from scipy.ndimage.morphology import binary_fill_holes
import skimage

import mesh2sdf.core

FUTURE3D_DIR = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'
FUTURE3D_METADATA = '/cluster/gondor/mdahnert/datasets/future3d/model_info.json'


def fix_meshes(num_proc=1, proc=0):

    with open(FUTURE3D_METADATA, 'r') as fin:
        future3d_metadata_ = json.load(fin)
    future3d_metadata = {}
    for entry in future3d_metadata_:
        future3d_metadata[entry['model_id']] = entry

    target_entries = [x for x in future3d_metadata if future3d_metadata[x]['super-category'] == 'Table'] # Sofa, Bed, Cabinet/Shelf/Desk, Lighting, Table
    target_entries = sorted(target_entries)
    target_entries = [x for i, x in enumerate(target_entries) if i % num_proc == proc]

    for obj_id in tqdm(target_entries):

        if os.path.exists(os.path.join(FUTURE3D_DIR, obj_id, 'raw_model.obj')):
            mesh = trimesh.load(os.path.join(FUTURE3D_DIR, obj_id, 'raw_model.obj'))
            vert_max = mesh.vertices.max()
            mesh.vertices = mesh.vertices / vert_max / 1.05
            sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, 320)

            binary_sdf_x = np.zeros_like(sdf)
            for i in range(sdf.shape[0]):
                binary_sdf_x[i, :, :] = binary_fill_holes(sdf[i, :, :] < 0)
            binary_sdf_x[sdf[:, :, :] < 0] = 0

            binary_sdf_y = np.zeros_like(sdf)
            for i in range(sdf.shape[1]):
                binary_sdf_y[:, i, :] = binary_fill_holes(sdf[:, i, :] < 0)
            binary_sdf_y[sdf[:, :, :] < 0] = 0

            binary_sdf_z = np.zeros_like(sdf)
            for i in range(sdf.shape[2]):
                binary_sdf_z[:, :, i] = binary_fill_holes(sdf[:, :, i] < 0)
            binary_sdf_z[sdf[:, :, :] < 0] = 0
            sdf[binary_sdf_x == 1] = -1
            sdf[binary_sdf_y == 1] = -1
            sdf[binary_sdf_z == 1] = -1

            level = 1. / 320
            vertices, faces, _, _ = skimage.measure.marching_cubes(sdf, level)
            faces = np.vstack([faces, faces[:, ::-1]])
            mesh_new = trimesh.Trimesh(vertices, faces)
            mesh_new.vertices = mesh_new.vertices / 320 * 2 - 1
            mesh_new.vertices = mesh_new.vertices * 1.05 * vert_max

            mesh_new.export(os.path.join(FUTURE3D_DIR, obj_id, 'raw_model_fixed.obj'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    fix_meshes(args.num_proc, args.proc)