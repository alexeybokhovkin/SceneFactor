import os, sys
import numpy as np
import json
import trimesh
from tqdm import tqdm

import mesh2sdf.core
from scipy.spatial import cKDTree


FRONT3DMESHESMANIFOLD = '/cluster/falas/abokhovkin/data/Front3D/manifold_meshes'
SAVEDIR = '/cluster/daidalos/abokhovkin/Front3D/chunked_data_lowres'
os.makedirs(SAVEDIR, exist_ok=True)


cat_to_id = {'Bed': 2,
             'Pier/Stool': 3,
             'Cabinet/Shelf/Desk': 4,
             'Lighting': 5,
             'Sofa': 6,
             'Chair': 7,
             'Table': 8,
             'Others': 9}


def put_scene_to_center(vertices, cell_center, cell_size):
    '''
        vertices: vertices of a mesh to move to origin and rescale 
        cell_center: original center of an input chunk
        cell_size: original size of an input chunk
    '''
    vertices = vertices - cell_center
    vertices = vertices / float(cell_size / 2)
    return vertices


def put_scene_back_from_center(vertices, cell_center, cell_size):
    '''
    reverse function to "put_scene_to_center"
    '''
    vertices = vertices * cell_size / 2
    vertices = vertices + cell_center
    return vertices


def compute_sdf_chunk(vertices, faces, num_voxels, chunk_size):
    sdf = mesh2sdf.core.compute(vertices, faces, num_voxels)
    xyz_grid = np.zeros((num_voxels, num_voxels, num_voxels, 3))
    dx = chunk_size / num_voxels
    for i in range(num_voxels):
        for j in range(num_voxels):
            for k in range(num_voxels):
                xyz_grid[i, j, k] = np.array([-1. + i * dx, -1. + j * dx, -1. + k * dx])
    xyz_grid_flattened = xyz_grid.reshape((-1, 3))
    sdf_new_flattened = sdf.reshape((-1))
    return xyz_grid_flattened, sdf_new_flattened, sdf


def compute_chunks(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(FRONT3DMESHESMANIFOLD))[:]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

    with open('valid_scenes.json', 'r') as fin:
        valid_scenes = json.load(fin)['valid_scenes']

    chunk_size = 3.8
    trunc_dist = 0.08
    scale_margin = 1.0
    max_coord_abs = 1. / scale_margin
    starting_point = mesh.bounds[0] - 0.14
    # 90 voxels for lowres, 180 voxels for highres
    num_voxels = 90 # 90 / 180
    num_trunc_points = 200000


    for obj_id in tqdm(all_obj_ids):
        try:

            if obj_id not in valid_scenes:
                continue

            LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
            # if os.path.exists(LOCAL_SAVEDIR):
            #     continue
            os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

            try:
                mesh_scene = trimesh.load(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'scene.obj'))
                all_mesh_furniture_filenames = [x for x in os.listdir(os.path.join(FRONT3DMESHESMANIFOLD, obj_id)) if x.endswith('_furniture.obj')]
                all_mesh_furniture = [trimesh.load(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, x)) for x in all_mesh_furniture_filenames]
                if len(all_mesh_furniture) == 0:
                    continue
                mesh_furniture = trimesh.util.concatenate(all_mesh_furniture)
                mesh = trimesh.util.concatenate([mesh_scene, mesh_furniture])
            except:
                continue

            all_scene_pts = []
            with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'scene_points.json'), 'r') as fin:
                scene_pts = json.load(fin)
                for key in scene_pts:
                    all_scene_pts += scene_pts[key]
            all_furniture_pts = []
            all_furniture_pts_canonic = []
            all_furniture_sem_ids = []
            all_furniture_inst_ids = []
            id_inst = 1

            try:
                with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'furniture_points.json'), 'r') as fin:
                    furniture_pts = json.load(fin)
                with open(os.path.join(FRONT3DMESHESMANIFOLD, obj_id, 'furniture_points_canonic.json'), 'r') as fin:
                    furniture_canonic_pts = json.load(fin)
                for key in furniture_pts:
                    all_furniture_pts += furniture_pts[key][0]
                    all_furniture_pts_canonic += furniture_canonic_pts[key][0]
                    all_furniture_sem_ids += [cat_to_id[furniture_pts[key][1]] for _ in range(len(furniture_pts[key][0]))]
                    all_furniture_inst_ids += [id_inst for _ in range(len(furniture_pts[key][0]))]
                    id_inst += 1
                all_scene_pts = np.array(all_scene_pts)
                all_furniture_pts = np.array(all_furniture_pts)
                all_furniture_pts_canonic = np.array(all_furniture_pts_canonic)
                all_furniture_sem_ids = np.array(all_furniture_sem_ids)
                all_furniture_inst_ids = np.array(all_furniture_inst_ids)
            except FileNotFoundError:
                continue

            try:
                furniture_points = np.array(trimesh.sample.sample_surface(mesh_furniture, 5000)[0])
            except:
                print(obj_id)
                print(len(all_mesh_furniture))
                continue

            first_center_point = starting_point + chunk_size / 2.
            n_x = int(np.abs(mesh.bounds[0][0] - mesh.bounds[1][0]) // chunk_size + 1)
            n_y = int(np.abs(mesh.bounds[0][1] - mesh.bounds[1][1]) // chunk_size + 1)
            n_z = int(np.abs(mesh.bounds[0][2] - mesh.bounds[1][2]) // chunk_size + 1)
            x_centers = np.array([first_center_point + [i * chunk_size, 0, 0] for i in range(n_x)])
            xy_centers = np.array([x_centers + [0, i * chunk_size, 0] for i in range(n_y)])
            xyz_centers = np.array([xy_centers + [0, 0, i * chunk_size] for i in range(n_z)])
            xyz_centers = np.reshape(xyz_centers, (-1, 1, 3))

            xyz_indices = np.array([[[(i, j, k) for i in range(n_x)] for j in range(n_y)] for k in range(n_z)])
            xyz_indices = np.reshape(xyz_indices, (-1, 3))

            # compute cell bounds to split chunks
            plane_offsets = np.array([[1, 0, 0],
                                      [-1, 0, 0],
                                      [0, 1, 0],
                                      [0, -1, 0],
                                      [0, 0, 1],
                                      [0, 0, -1]]).reshape(1, 6, 3)
            plane_offsets = plane_offsets * (scale_margin * chunk_size / 2.)
            plane_origins = xyz_centers + plane_offsets
            cell_bounds = [np.array([[x[1][0], x[3][1], x[5][2]], 
                                     [x[0][0], x[2][1], x[4][2]]]) for x in plane_origins]
            cell_bounds = np.array(cell_bounds)

            resulting_full_chunk_size_m = chunk_size * scale_margin
            resulting_inner_chunk_size_m = chunk_size
            resulting_resolution_m = resulting_full_chunk_size_m / float(num_voxels)

            
            for i, cell_bound in enumerate(cell_bounds):
                xyz_index = xyz_indices[i]
                index_filename = '_'.join([str(k) for k in xyz_index])

                if xyz_index[1] != 0:
                    continue

                cell_center = (cell_bound[0] + cell_bound[1]) / 2.
                mesh.vertices = put_scene_to_center(mesh.vertices, cell_center, resulting_full_chunk_size_m)
                furniture_points = put_scene_to_center(furniture_points, cell_center, resulting_full_chunk_size_m)

                all_scene_pts[:, :3] = put_scene_to_center(all_scene_pts[:, :3], cell_center, resulting_full_chunk_size_m)
                all_furniture_pts = put_scene_to_center(all_furniture_pts, cell_center, resulting_full_chunk_size_m)
                all_scene_pts_chunk_ids = np.where((-1.5 <= all_scene_pts[:, 0]) & (-1.5 <= all_scene_pts[:, 1]) & (-1.5 <= all_scene_pts[:, 2]) & \
                                               (1.5 >= all_scene_pts[:, 0]) & (1.5 >= all_scene_pts[:, 1]) & (1.5 >= all_scene_pts[:, 2]))[0]
                all_furniture_pts_chunk_ids = np.where((-1.5 <= all_furniture_pts[:, 0]) & (-1.5 <= all_furniture_pts[:, 1]) & (-1.5 <= all_furniture_pts[:, 2]) & \
                                                   (1.5 >= all_furniture_pts[:, 0]) & (1.5 >= all_furniture_pts[:, 1]) & (1.5 >= all_furniture_pts[:, 2]))[0]
                scene_pts_chunk = all_scene_pts[all_scene_pts_chunk_ids]
                furniture_pts_chunk = all_furniture_pts[all_furniture_pts_chunk_ids]
                furniture_canonic_pts_chunk = all_furniture_pts_canonic[all_furniture_pts_chunk_ids]
                furniture_sem_ids_chunk = all_furniture_sem_ids[all_furniture_pts_chunk_ids]
                furniture_inst_ids_chunk = all_furniture_inst_ids[all_furniture_pts_chunk_ids]

                # print('Computing chunk SDF')
                xyz_grid, sdf_grid, sdf_tensor = compute_sdf_chunk(mesh.vertices, mesh.faces, num_voxels, chunk_size)

                # print('Truncating SDF')
                inside_chunk_ids = np.where((xyz_grid[:, 0] >= -max_coord_abs) & (xyz_grid[:, 1] >= -max_coord_abs) & (xyz_grid[:, 2] >= -max_coord_abs) & \
                                            (xyz_grid[:, 0] < max_coord_abs) & (xyz_grid[:, 1] < max_coord_abs) & (xyz_grid[:, 2] < max_coord_abs))[0]
                xyz_grid_inside = xyz_grid[inside_chunk_ids]
                sdf_grid_inside = sdf_grid[inside_chunk_ids]

                inside_chunk_furniture_ids = np.where((furniture_points[:, 0] >= -max_coord_abs) & (furniture_points[:, 1] >= -max_coord_abs) & (furniture_points[:, 2] >= -max_coord_abs) & \
                                                      (furniture_points[:, 0] < max_coord_abs) & (furniture_points[:, 1] < max_coord_abs) & (furniture_points[:, 2] < max_coord_abs))[0]
                flag_furniture = True if len(inside_chunk_furniture_ids) > 0 else False

                non_truncation_ids = np.where(sdf_grid_inside <= trunc_dist)[0]
                xyz_grid_non_trunc = xyz_grid_inside[non_truncation_ids]
                sdf_grid_non_trunc = sdf_grid_inside[non_truncation_ids]

                truncation_ids = np.where(sdf_grid_inside > trunc_dist)[0]
                xyz_grid_trunc = xyz_grid_inside[truncation_ids]
                sdf_grid_trunc = sdf_grid_inside[truncation_ids]

                xyz_grid_non_trunc = put_scene_back_from_center(xyz_grid_non_trunc, cell_center, resulting_full_chunk_size_m)
                xyz_grid_trunc = put_scene_back_from_center(xyz_grid_trunc, cell_center, resulting_full_chunk_size_m)
                mesh.vertices = put_scene_back_from_center(mesh.vertices, cell_center, resulting_full_chunk_size_m)
                furniture_points = put_scene_back_from_center(furniture_points, cell_center, resulting_full_chunk_size_m)
                xyz_sdf = np.hstack([xyz_grid_non_trunc, sdf_grid_non_trunc[..., None]])
                xyz_sdf = xyz_sdf.astype('float32')

                all_scene_pts[:, :3] = put_scene_back_from_center(all_scene_pts[:, :3], cell_center, resulting_full_chunk_size_m)
                all_furniture_pts = put_scene_back_from_center(all_furniture_pts, cell_center, resulting_full_chunk_size_m)

                xyz_sdf_trunc = np.hstack([xyz_grid_trunc, sdf_grid_trunc[..., None]])
                xyz_sdf_trunc[:, 3] = trunc_dist
                xyz_sdf_trunc = xyz_sdf_trunc.astype('float32')
                random_indices = np.random.choice(len(xyz_sdf_trunc), min(len(xyz_sdf_trunc), num_trunc_points), replace=False)
                xyz_sdf_trunc = xyz_sdf_trunc[random_indices]

                scene_pts_chunk = np.hstack([scene_pts_chunk, np.ones((len(scene_pts_chunk), 1)) * 1, np.zeros((len(scene_pts_chunk), 4))])
                # furniture_pts_chunk = np.hstack([furniture_pts_chunk, np.ones((len(furniture_pts_chunk), 1)) * 2])
                furniture_pts_chunk = np.hstack([furniture_pts_chunk, 
                                                 furniture_sem_ids_chunk[:, None], 
                                                 furniture_inst_ids_chunk[:, None],
                                                 furniture_canonic_pts_chunk])
                all_pts_chunk = np.vstack([scene_pts_chunk, furniture_pts_chunk])

                try:
                    dx = 2.0 / num_voxels
                    xyz_grid = np.zeros((num_voxels, num_voxels, num_voxels, 3))
                    for i in range(num_voxels):
                        for j in range(num_voxels):
                            for k in range(num_voxels):
                                xyz_grid[i, j, k] = np.array([-1. + i * dx, -1. + j * dx, -1. + k * dx])
                    xyz_grid_flattened = xyz_grid.reshape((-1, 3))
                    xyzd = np.hstack([xyz_grid_flattened, sdf_tensor.reshape((-1))[..., None]])

                    non_trunc_points = xyzd[:, :3][np.abs(xyzd[:, 3]) < 0.07] # 0.02
                    non_trunc_points_ids = np.where(np.abs(xyzd[:, 3]) < 0.07)[0] # 0.02


                    non_trunc_points_sem = xyzd[:, :3][np.abs(xyzd[:, 3]) < 0.05] # 0.02
                    non_trunc_points_ids_sem = np.where(np.abs(xyzd[:, 3]) < 0.05)[0] # 0.02

                    tree = cKDTree(all_pts_chunk[:, :3])
                    dists, idx = tree.query(non_trunc_points_sem)

                    voxel_distribution = all_pts_chunk[idx][:, 3:5]
                    all_voxel_distribution_sem = np.zeros((len(xyz_grid_flattened)))
                    all_voxel_distribution_sem[non_trunc_points_ids_sem] = voxel_distribution[:, 0]
                    all_voxel_distribution_sem = all_voxel_distribution_sem.reshape((num_voxels, num_voxels, num_voxels)).astype('int16')
                    all_voxel_distribution_inst = np.zeros((len(xyz_grid_flattened)))
                    all_voxel_distribution_inst[non_trunc_points_ids_sem] = voxel_distribution[:, 1]
                    all_voxel_distribution_inst = all_voxel_distribution_inst.reshape((num_voxels, num_voxels, num_voxels)).astype('int16')

                    canonic_coords = all_pts_chunk[idx][:, 5:]
                    all_canonic_distribution = np.zeros((len(xyz_grid_flattened), 3))
                    all_canonic_distribution[non_trunc_points_ids_sem] = canonic_coords[:]
                    all_canonic_distribution = (all_canonic_distribution.reshape((num_voxels, num_voxels, num_voxels, 3)) * 256).astype('int16')
                except:
                    continue

                if len(non_truncation_ids) != 0:
                    meta_data = {
                        'chink_size': chunk_size,
                        'scale_margin': scale_margin,
                        'max_coord_abs': max_coord_abs,
                        'starting_point': starting_point.tolist(),
                        'num_voxels': num_voxels,
                        'trunc_dist': trunc_dist,
                        'furniture': flag_furniture,
                        'cell_center': cell_center.tolist(),
                        'cell_scale': resulting_full_chunk_size_m,
                        'subchunk_coords': cell_bound.tolist()
                    }
                    with open(os.path.join(LOCAL_SAVEDIR, f'{index_filename}.json'), 'w') as fout:
                        json.dump(meta_data, fout)

                    np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}.npy'), sdf_tensor)
                    np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_semantic.npy'), all_voxel_distribution_sem)
                    np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_instance.npy'), all_voxel_distribution_inst)
                    np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_canonic.npy'), all_canonic_distribution)
                    
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_scene_pts.npy'), scene_pts_chunk.astype('float16'))
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_furniture_pts.npy'), furniture_pts_chunk.astype('float16'))

                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_all_pts.npy'), all_pts_chunk)
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{index_filename}_non_trunc_pts.npy'), non_trunc_points_sem)

        except FileNotFoundError:
            continue



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_chunks(args.num_proc, args.proc)