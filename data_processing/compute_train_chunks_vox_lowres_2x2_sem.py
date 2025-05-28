import os, sys
import numpy as np
import json
from tqdm import tqdm
from copy import deepcopy


FRONT3D_DATA = '/cluster/daidalos/abokhovkin/Front3D/chunked_data_lowres'
SAVEDIR = '/cluster/andram/abokhovkin/data/Front3D/train_chunks_sem_lowres_2x2_sem_canonic'
os.makedirs(SAVEDIR, exist_ok=True)


def compute_chunks(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(FRONT3D_DATA))[:]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc and not x.endswith('.json')]

    for obj_id in tqdm(all_obj_ids):
        try:

            LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
            # if os.path.exists(LOCAL_SAVEDIR):
            #     continue

            chunk_size = 90
            subchunk_size = 64 # 2x scaling
            num_subchunks = 64

            # pick only chunks with furniture
            json_files = [x for x in os.listdir(os.path.join(FRONT3D_DATA, obj_id)) if x.endswith('.json')]
            scene_map = {}
            for json_file in json_files:
                with open(os.path.join(FRONT3D_DATA, obj_id, json_file), 'r') as fin:
                    meta_data = json.load(fin)
                scene_map[json_file.split('.')[0]] = meta_data['furniture']
            scene_map_keys = list(scene_map.keys())
            scene_map_keys_furniture = [x for x in scene_map_keys if scene_map[x] is True]
            scene_map_keys_furniture_nn = {x: [x] for x in scene_map_keys_furniture}
            for chunk_id in scene_map_keys_furniture:
                chunk_tokens = [int(x) for x in chunk_id.split('_')]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] - 1), '0', str(chunk_tokens[2] - 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] - 1), '0', str(chunk_tokens[2])])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] - 1), '0', str(chunk_tokens[2] + 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0]), '0', str(chunk_tokens[2] - 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0]), '0', str(chunk_tokens[2] + 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] + 1), '0', str(chunk_tokens[2] - 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] + 1), '0', str(chunk_tokens[2])])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
                chunk_id_nn = '_'.join([str(chunk_tokens[0] + 1), '0', str(chunk_tokens[2] + 1)])
                if chunk_id_nn in scene_map_keys:
                    scene_map_keys_furniture_nn[chunk_id] += [chunk_id_nn]
            
            # sample chunks
            for chunk_id in scene_map_keys_furniture_nn:
                
                all_chunk_bounds = []

                big_tensor = np.ones((chunk_size * 3, chunk_size, chunk_size * 3))
                central_chunk = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id}.npy'))
                big_tensor[chunk_size:chunk_size * 2, :, chunk_size:chunk_size * 2] = central_chunk
                chunk_id_tokens = [int(x) for x in chunk_id.split('_')]

                with open(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id}.json'), 'r') as fin:
                    chunk_meta_data = json.load(fin)
                    chunk_cell_bounds = chunk_meta_data['cell_bound']
                    all_chunk_bounds += [chunk_cell_bounds]

                big_tensor_sem = np.zeros((chunk_size * 3, chunk_size, chunk_size * 3))
                central_chunk_sem = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id}_semantic.npy'))
                big_tensor_sem[chunk_size:chunk_size * 2, :, chunk_size:chunk_size * 2] = central_chunk_sem

                big_tensor_inst = np.zeros((chunk_size * 3, chunk_size, chunk_size * 3))
                central_chunk_inst = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id}_instance.npy'))
                big_tensor_inst[chunk_size:chunk_size * 2, :, chunk_size:chunk_size * 2] = central_chunk_inst

                big_tensor_canonic = np.zeros((chunk_size * 3, chunk_size, chunk_size * 3, 3))
                central_chunk_canonic = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id}_canonic.npy'))
                big_tensor_canonic[chunk_size:chunk_size * 2, :, chunk_size:chunk_size * 2] = central_chunk_canonic
                
                for chunk_id_nn in scene_map_keys_furniture_nn[chunk_id][:]:
                    central_chunk_nn = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id_nn}.npy'))
                    chunk_id_nn_tokens = [int(x) for x in chunk_id_nn.split('_')]
                    offset_x = chunk_id_tokens[0] - chunk_id_nn_tokens[0]
                    offset_z = chunk_id_tokens[2] - chunk_id_nn_tokens[2]
                    big_tensor[chunk_size - chunk_size * offset_x:2 * chunk_size - chunk_size * offset_x, :, chunk_size - chunk_size * offset_z:2 * chunk_size - chunk_size * offset_z] = central_chunk_nn
                
                    central_chunk_nn_sem = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id_nn}_semantic.npy'))
                    big_tensor_sem[chunk_size - chunk_size * offset_x:2 * chunk_size - chunk_size * offset_x, :, chunk_size - chunk_size * offset_z:2 * chunk_size - chunk_size * offset_z] = central_chunk_nn_sem

                    central_chunk_nn_inst = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id_nn}_instance.npy'))
                    big_tensor_inst[chunk_size - chunk_size * offset_x:2 * chunk_size - chunk_size * offset_x, :, chunk_size - chunk_size * offset_z:2 * chunk_size - chunk_size * offset_z] = central_chunk_nn_inst

                    central_chunk_nn_canonic = np.load(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id_nn}_canonic.npy'))
                    big_tensor_canonic[chunk_size - chunk_size * offset_x:2 * chunk_size - chunk_size * offset_x, :, chunk_size - chunk_size * offset_z:2 * chunk_size - chunk_size * offset_z] = central_chunk_nn_canonic

                    with open(os.path.join(FRONT3D_DATA, obj_id, f'{chunk_id_nn}.json'), 'r') as fin:
                        chunk_meta_data = json.load(fin)
                        chunk_cell_bounds = chunk_meta_data['cell_bound']
                        all_chunk_bounds += [chunk_cell_bounds]

                all_chunk_bounds = np.array(all_chunk_bounds)
                big_tensor_bounds = np.array([[np.min(all_chunk_bounds[:, 0, 0]), np.min(all_chunk_bounds[:, 0, 1]), np.min(all_chunk_bounds[:, 0, 2])],
                                              [np.max(all_chunk_bounds[:, 1, 0]), np.max(all_chunk_bounds[:, 1, 1]), np.max(all_chunk_bounds[:, 1, 2])]])

                info_coords = np.where((big_tensor < 0.5))
                
                # valid_range = [[info_coords[0].min() + chunk_size / 2., info_coords[1].min(), info_coords[2].min() + chunk_size / 2.],
                #                [info_coords[0].max() - chunk_size / 2., info_coords[1].min(), info_coords[2].max() - chunk_size / 2.]]
                valid_range = [[info_coords[0].min(), info_coords[1].min(), info_coords[2].min()],
                               [info_coords[0].max(), info_coords[1].min(), info_coords[2].max()]]
                valid_range = np.array(valid_range).astype('int32')
                valid_range_min_point = [[valid_range[0][0], valid_range[0][1], valid_range[0][2]],
                                         [max(valid_range[0][0], valid_range[1][0] - 2 * subchunk_size), max(valid_range[0][1], valid_range[1][1] - subchunk_size), max(valid_range[0][2], valid_range[1][2] - 2 * subchunk_size)]]
                
                all_coords = np.where((big_tensor < 0.5))
                all_coords_range = np.array([[all_coords[0].min(), all_coords[1].min(), all_coords[2].min()],
                                             [all_coords[0].max(), all_coords[1].max(), all_coords[2].max()]])
                
                all_subchunk_bounds = []
                all_subchunk_coord_bounds = []
                for i in range(num_subchunks):
                    # print(valid_range_min_point)
                    random_min_point = np.random.uniform(low=valid_range_min_point[0], high=valid_range_min_point[1], size=(3,)).astype('int32')
                    random_max_point = random_min_point + np.array([2 * subchunk_size, subchunk_size, 2 * subchunk_size])
                    subchunk = big_tensor[random_min_point[0]:random_max_point[0],
                                          random_min_point[1]:random_max_point[1],
                                          random_min_point[2]:random_max_point[2]]
                    subchunk_sem = big_tensor_sem[random_min_point[0]:random_max_point[0],
                                                  random_min_point[1]:random_max_point[1],
                                                  random_min_point[2]:random_max_point[2]]
                    subchunk_inst = big_tensor_inst[random_min_point[0]:random_max_point[0],
                                                  random_min_point[1]:random_max_point[1],
                                                  random_min_point[2]:random_max_point[2]]
                    subchunk_canonic = big_tensor_canonic[random_min_point[0]:random_max_point[0],
                                                          random_min_point[1]:random_max_point[1],
                                                          random_min_point[2]:random_max_point[2]]
                    
                    subchunk_ratios = np.array([[(random_min_point[0] - all_coords_range[0][0]) / (all_coords_range[1][0] - all_coords_range[0][0]),
                                                 (random_min_point[1] - all_coords_range[0][1]) / (all_coords_range[1][1] - all_coords_range[0][1]),
                                                 (random_min_point[2] - all_coords_range[0][2]) / (all_coords_range[1][2] - all_coords_range[0][2])],

                                                [(random_max_point[0] - all_coords_range[0][0]) / (all_coords_range[1][0] - all_coords_range[0][0]),
                                                 (random_max_point[1] - all_coords_range[0][1]) / (all_coords_range[1][1] - all_coords_range[0][1]),
                                                 (random_max_point[2] - all_coords_range[0][2]) / (all_coords_range[1][2] - all_coords_range[0][2])]])
                    
                    subchunk_coord_bounds = np.array([[big_tensor_bounds[0][0] + subchunk_ratios[0][0] * (big_tensor_bounds[1][0] - big_tensor_bounds[0][0]),
                                                       big_tensor_bounds[0][1] + subchunk_ratios[0][1] * (big_tensor_bounds[1][1] - big_tensor_bounds[0][1]),
                                                       big_tensor_bounds[0][2] + subchunk_ratios[0][2] * (big_tensor_bounds[1][2] - big_tensor_bounds[0][2])],
                                                      
                                                      [big_tensor_bounds[0][0] + subchunk_ratios[1][0] * (big_tensor_bounds[1][0] - big_tensor_bounds[0][0]),
                                                       big_tensor_bounds[0][1] + subchunk_ratios[1][1] * (big_tensor_bounds[1][1] - big_tensor_bounds[0][1]),
                                                       big_tensor_bounds[0][2] + subchunk_ratios[1][2] * (big_tensor_bounds[1][2] - big_tensor_bounds[0][2])]])
                    
                    num_furniture_voxels = len(np.where(subchunk_sem >= 2)[0])
                    if num_furniture_voxels >= 5000: # 10000
                        subchunk_bounds = np.array([[random_min_point[0], random_min_point[1], random_min_point[2]],
                                                    [random_max_point[0], random_max_point[1], random_max_point[2]]])
                        all_subchunk_bounds += [subchunk_bounds]
                        all_subchunk_coord_bounds += [subchunk_coord_bounds]

                if len(all_subchunk_bounds) != 0:
                    all_subchunk_bounds = np.array(all_subchunk_bounds)
                    subchunks_together_bounds = np.array([[np.min(all_subchunk_bounds[:, 0, 0]), np.min(all_subchunk_bounds[:, 0, 1]), np.min(all_subchunk_bounds[:, 0, 2])],
                                                            [np.max(all_subchunk_bounds[:, 1, 0]), np.max(all_subchunk_bounds[:, 1, 1]), np.max(all_subchunk_bounds[:, 1, 2])]])

                    LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
                    os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

                    big_subchunk = big_tensor[subchunks_together_bounds[0][0]:subchunks_together_bounds[1][0],
                                                subchunks_together_bounds[0][1]:subchunks_together_bounds[1][1],
                                                subchunks_together_bounds[0][2]:subchunks_together_bounds[1][2]]
                    big_subchunk_sem = big_tensor_sem[subchunks_together_bounds[0][0]:subchunks_together_bounds[1][0],
                                                        subchunks_together_bounds[0][1]:subchunks_together_bounds[1][1],
                                                        subchunks_together_bounds[0][2]:subchunks_together_bounds[1][2]]
                    big_subchunk_inst = big_tensor_inst[subchunks_together_bounds[0][0]:subchunks_together_bounds[1][0],
                                                        subchunks_together_bounds[0][1]:subchunks_together_bounds[1][1],
                                                        subchunks_together_bounds[0][2]:subchunks_together_bounds[1][2]]
                    big_subchunk_canonic = big_tensor_canonic[subchunks_together_bounds[0][0]:subchunks_together_bounds[1][0],
                                                              subchunks_together_bounds[0][1]:subchunks_together_bounds[1][1],
                                                              subchunks_together_bounds[0][2]:subchunks_together_bounds[1][2]]
                    
                    unique_inst_ids = np.unique(big_subchunk_inst)
                    big_subchunk_sem_buf = deepcopy(big_subchunk_sem)
                    big_subchunk_sem_rough = deepcopy(big_subchunk_sem)
                    big_subchunk_canonic_buf = deepcopy(big_subchunk_canonic)
                    for inst_id in unique_inst_ids:
                        if inst_id < 1:
                            continue
                        inst_coords = np.where(big_subchunk_inst == inst_id)

                        inst_coords_vec = np.vstack(inst_coords).T
                        min_coord = np.min(inst_coords_vec, axis=0)
                        max_coord = np.max(inst_coords_vec, axis=0)
                        colors_flat = big_subchunk_canonic[inst_coords_vec[:, 0], inst_coords_vec[:, 1], inst_coords_vec[:, 2]]
                        min_id = [np.argmin(colors_flat[:, 0]), np.argmin(colors_flat[:, 1]), np.argmin(colors_flat[:, 2])]
                        max_id = [np.argmax(colors_flat[:, 0]), np.argmax(colors_flat[:, 1]), np.argmax(colors_flat[:, 2])]
                        min_color_id = [inst_coords_vec[min_id[0], 0], inst_coords_vec[min_id[1], 1], inst_coords_vec[min_id[2], 2]]
                        max_color_id = [inst_coords_vec[max_id[0], 0], inst_coords_vec[max_id[1], 1], inst_coords_vec[max_id[2], 2]]

                        color_for_min_coord = []
                        color_for_max_coord = []
                        for i in range(len(min_coord)):
                            if np.abs(min_coord[i] - min_color_id[i]) < np.abs(min_coord[i] - max_color_id[i]):
                                color_for_min_coord += [colors_flat[min_id[i], i]]
                                color_for_max_coord += [colors_flat[max_id[i], i]]
                            else:
                                color_for_min_coord += [colors_flat[max_id[i], i]]
                                color_for_max_coord += [colors_flat[min_id[i], i]]

                        color_for_min_coord = np.array(color_for_min_coord)
                        color_for_max_coord = np.array(color_for_max_coord)
                        diff_coords = max_coord - min_coord
                        if diff_coords[0] == 0:
                            diff_coords[0] = 1
                        if diff_coords[1] == 0:
                            diff_coords[1] = 1
                        if diff_coords[2] == 0:
                            diff_coords[2] = 1
                        inc_direction = (color_for_max_coord - color_for_min_coord) / diff_coords
                        
                        for ii, i in enumerate(range(min_coord[0], max_coord[0] + 1)):
                            for jj, j in enumerate(range(min_coord[1], max_coord[1] + 1)):
                                for kk, k in enumerate(range(min_coord[2], max_coord[2] + 1)):
                                    big_subchunk_canonic_buf[i, j, k] = np.array([int(color_for_min_coord[0] + inc_direction[0] * ii), 
                                                                              int(color_for_min_coord[1] + inc_direction[1] * jj), 
                                                                              int(color_for_min_coord[2] + inc_direction[2] * kk)])
                                    
                                    if i == max_coord[0] and j == max_coord[1] and k == max_coord[2]:
                                        max_act_color = big_subchunk_canonic_buf[i, j, k] 
                        min_act_color = np.array([int(color_for_min_coord[0] + inc_direction[0] * 0), 
                                                                              int(color_for_min_coord[1] + inc_direction[1] * 0), 
                                                                              int(color_for_min_coord[2] + inc_direction[2] * 0)])
            
                        if len(inst_coords) != 0:
                            counts = np.bincount(big_subchunk_sem[inst_coords[0], inst_coords[1], inst_coords[2]].astype('int32'))
                            sem_label = np.argmax(counts)

                            # enable if you want make 3D boxes
                            big_subchunk_sem_buf[np.min(inst_coords[0]):np.max(inst_coords[0]),
                                                 np.min(inst_coords[1]):np.max(inst_coords[1]),
                                                 np.min(inst_coords[2]):np.max(inst_coords[2])] = sem_label
                    big_subchunk_sem = big_subchunk_sem_buf
                    big_subchunk_canonic = big_subchunk_canonic_buf

                    big_subchunk_canonic = np.clip(big_subchunk_canonic, 0, 255)
                            
                    np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}.npy'), big_subchunk)
                    np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_semantic.npy'), big_subchunk_sem.astype('int16'))
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_semanticrough.npy'), big_subchunk_sem_rough.astype('int16'))
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_canonic.npy'), big_subchunk_canonic.astype('int16'))
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_semantic_full.npy'), big_tensor_sem.astype('int16'))
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_instance_full.npy'), big_tensor_inst.astype('int16'))

                    for k in range(len(all_subchunk_coord_bounds)):
                        subchunk_within_chunk_bounds = np.array([[all_subchunk_bounds[k][0][0] - subchunks_together_bounds[0][0],
                                                                    all_subchunk_bounds[k][0][1] - subchunks_together_bounds[0][1],
                                                                    all_subchunk_bounds[k][0][2] - subchunks_together_bounds[0][2]],
                                                                    
                                                                    [all_subchunk_bounds[k][1][0] - subchunks_together_bounds[0][0],
                                                                    all_subchunk_bounds[k][1][1] - subchunks_together_bounds[0][1],
                                                                    all_subchunk_bounds[k][1][2] - subchunks_together_bounds[0][2]]])

                        chunk_meta_data = {}
                        chunk_meta_data['subchunk_bounds'] = subchunk_within_chunk_bounds.tolist()
                        chunk_meta_data['subchunk_coords'] = all_subchunk_coord_bounds[k].tolist()
                        chunk_meta_data['subchunks_together_bounds'] = subchunks_together_bounds.tolist()
                        chunk_meta_data['big_chunk_bounds'] = all_coords_range.tolist()

                        with open(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_{k}.json'), 'w') as fout:
                            json.dump(chunk_meta_data, fout)
                
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_{i}.npy'), subchunk)
                    # np.save(os.path.join(LOCAL_SAVEDIR, f'{chunk_id}_{i}_semantic.npy'), subchunk_sem.astype('int16'))
        except FileNotFoundError:
            continue


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_chunks(args.num_proc, args.proc)