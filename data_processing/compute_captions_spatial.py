import os, sys
import numpy as np
import json
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import random
import re
from copy import deepcopy
from scipy.spatial import cKDTree


# CHUNKS_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference' # inference chunks
CHUNKS_DIR = '/cluster/andram/abokhovkin/data/Front3D/train_chunks_sem_lowres_2x2_sem_canonic' # training chunks

FRONT3DJSON = '/cluster/gondor/mdahnert/datasets/front3d/3D-FRONT'
FUTURE3DMODEL = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'

vowels = ['a', 'u', 'i', 'e', 'o', 'y']


np.random.seed(174)


def check_mesh_inside(points, bounds):
    points_inside = [1 for point in points if (bounds[0][0] <= point[0] <= bounds[1][0]) and (bounds[0][1] <= point[1] <= bounds[1][1]) and (bounds[0][2] <= point[2] <= bounds[1][2])]
    points_inside_explicit = [point for point in points if (bounds[0][0] <= point[0] <= bounds[1][0]) and (bounds[0][1] <= point[1] <= bounds[1][1]) and (bounds[0][2] <= point[2] <= bounds[1][2])]
    if len(points_inside_explicit) > 0:
        points_inside_explicit = np.vstack(points_inside_explicit)
    else:
        points_inside_explicit = []
    num_all_points = len(points)
    if num_all_points == 0:
        return 0, points_inside_explicit
    else:
        num_points_inside = np.sum(points_inside)
        return num_points_inside / num_all_points > 0.35, points_inside_explicit
    
def check_mesh_inside_thr(points, bounds):
    points_inside = [1 for point in points if (bounds[0][0] <= point[0] <= bounds[1][0]) and (bounds[0][1] <= point[1] <= bounds[1][1]) and (bounds[0][2] <= point[2] <= bounds[1][2])]
    points_inside_explicit = [point for point in points if (bounds[0][0] <= point[0] <= bounds[1][0]) and (bounds[0][1] <= point[1] <= bounds[1][1]) and (bounds[0][2] <= point[2] <= bounds[1][2])]
    if len(points_inside_explicit) > 0:
        points_inside_explicit = np.vstack(points_inside_explicit)
    else:
        points_inside_explicit = []
    num_all_points = len(points)
    if num_all_points == 0:
        return 0
    else:
        num_points_inside = np.sum(points_inside)
        return num_points_inside / num_all_points
    
def get_wall_points_inside(points, bounds):
    points_inside = [point for point in points if (bounds[0][0] <= point[0] <= bounds[1][0]) and (bounds[0][1] <= point[1] <= bounds[1][1]) and (bounds[0][2] <= point[2] <= bounds[1][2])]
    if len(points_inside) > 0:
        points_inside_explicit = np.vstack(points_inside)
    else:
        points_inside_explicit = []
    return points_inside_explicit

def create_mesh_groups(all_points, dist_thr=0.3):
    all_edges = []
    for i in range(len(all_points)):
        for j in range(len(all_points)):
            if j > i:
                dist = cdist(all_points[i], all_points[j]).min()
                if dist <= dist_thr:
                    all_edges += [(i, j)]
    all_groups = []
    used_ids = []
    for i in range(len(all_points)):
        if i not in used_ids:
            cur_group = []
            updated_group = [i]
            while len(cur_group) != len(updated_group):
                cur_group = updated_group[:]
                for edge in all_edges:
                    if edge[0] in cur_group or edge[1] in cur_group:
                        updated_group += [edge[0]]
                        updated_group += [edge[1]]
                updated_group = list(set(updated_group))
                cur_group = list(set(cur_group))
            used_ids += updated_group
            all_groups += [updated_group]
    return all_groups

def compute_max_intersection_rate(range_1, range_2):
    if range_1[0] > range_2[0]:
        range_1, range_2 = range_2, range_1
    if range_1[1] <= range_2[0]:
        return 0.0
    else:
        intersection = range_1[1] - range_2[0]
        return max(intersection / (range_1[1] - range_1[0]), intersection / (range_2[1] - range_2[0]))

def compute_across_groups(all_groups, all_points):
    across_groups = []
    for i, group_i in enumerate(all_groups):
        for j, group_j in enumerate(all_groups):
            if i != j:
                group_i_pts, group_j_pts = [], []
                for k_i in group_i:
                    group_i_pts += [all_points[k_i]]
                for k_j in group_j:
                    group_j_pts += [all_points[k_j]]
                group_i_pts = np.array(group_i_pts)
                group_j_pts = np.array(group_j_pts)
                range_i_x = [np.min(group_i_pts[:, 0]), np.max(group_i_pts[:, 0])]
                range_i_y = [np.min(group_i_pts[:, 2]), np.max(group_i_pts[:, 2])]
                range_j_x = [np.min(group_j_pts[:, 0]), np.max(group_j_pts[:, 0])]
                range_j_y = [np.min(group_j_pts[:, 2]), np.max(group_j_pts[:, 2])]
                intersection_rate_x = compute_max_intersection_rate(range_i_x, range_j_x)
                intersection_rate_y = compute_max_intersection_rate(range_i_y, range_j_y)
                if intersection_rate_x >= 0.5 or intersection_rate_y >= 0.5:
                    across_groups += [(i, j)]
    return across_groups

def compute_close_to_wall(all_groups, all_points, wall_points, dist_thr=0.5):
    close_to_wall = []
    if len(wall_points) != 0:
        for i, group_i in enumerate(all_groups):
            group_i_pts = []
            for k_i in group_i:
                group_i_pts += [all_points[k_i]]
            if len(group_i_pts) != 0:
                group_i_pts = np.vstack(group_i_pts)
                dist = cdist(group_i_pts, wall_points).min()
                if dist <= dist_thr:
                    close_to_wall += [i]
    return close_to_wall

def add_wall_annotation(wall_points, chunk_bounds):

    def rotate_key(key, num_rots=0):
        keys = {
            'right': ['right', 'bottom', 'left', 'top'],
            'left': ['left', 'top', 'right', 'bottom'],
            'top': ['top', 'right', 'bottom', 'left'],
            'bottom': ['bottom', 'left', 'top', 'right']
        }
        if key == 'center':
            return key
        else:
            return keys[key][num_rots]
        

    chunk_bounds = np.array(chunk_bounds)
    min_point = chunk_bounds[0]
    scale = chunk_bounds[1] - chunk_bounds[0]

    wall_captions = []
    
    if len(wall_points) > 0:
        wall_points_centered = (wall_points - min_point) / scale
        points_right = [x for x in wall_points_centered if x[2] >= 0.7]
        points_left = [x for x in wall_points_centered if x[2] <= 0.3]
        points_top = [x for x in wall_points_centered if x[0] >= 0.7]
        points_bottom = [x for x in wall_points_centered if x[0] <= 0.3]
        points_center = [x for x in wall_points_centered if (x[0] > 0.3) & (x[0] < 0.7) & (x[2] > 0.3) & (x[2] < 0.7)]

        wall_anno = {
            'right': 0,
            'left': 0,
            'top': 0,
            'bottom': 0,
            'center': 0
        }
        if len(points_right) > 275:
            wall_anno['right'] = 1
        if len(points_left) > 275:
            wall_anno['left'] = 1
        if len(points_top) > 275:
            wall_anno['top'] = 1
        if len(points_bottom) > 275:
            wall_anno['bottom'] = 1
        if len(points_center) > 120:
            wall_anno['center'] = 1

        all_walls_anno = []
        for key in wall_anno:
            if wall_anno[key] == 1:
                all_walls_anno += [key]

        for k in range(4):
            if len(all_walls_anno) == 0:
                caption = ''
            else:
                caption = 'walls on the '
                for key in all_walls_anno:
                    rotated_key = rotate_key(key, k)
                    caption += (rotated_key + ', ')
                caption = caption[:-2]
            wall_captions += [caption]
    else:
        wall_captions = ['', '', '', '']
    return wall_captions


def add_layout_annotation(scene_points, chunk_bounds):
    
    def rotate_key(key, num_rots=0):
        keys = {
            'right': ['right', 'bottom', 'left', 'top'],
            'left': ['left', 'top', 'right', 'bottom'],
            'top': ['top', 'right', 'bottom', 'left'],
            'bottom': ['bottom', 'left', 'top', 'right'],
            'bottom right': ['bottom right', 'bottom left', 'top left', 'top right'],
            'top right': ['top right', 'bottom right', 'bottom left', 'top left'],
            'bottom left': ['bottom left', 'top left', 'top right', 'bottom right'],
            'top left': ['top left', 'top right', 'bottom right', 'bottom left']
        }
        if key == 'center':
            return key
        else:
            return keys[key][num_rots]
        
    chunk_bounds = np.array(chunk_bounds)
    min_point = chunk_bounds[0]
    scale = chunk_bounds[1] - chunk_bounds[0]
    
    min_thr = 0.05
    max_thr = 0.95
    
    if len(scene_points) > 0:
        scene_points_centered = (scene_points - min_point) / scale
        points_right = [x for x in scene_points_centered if x[2] >= max_thr]
        points_left = [x for x in scene_points_centered if x[2] <= min_thr]
        points_top = [x for x in scene_points_centered if x[0] >= max_thr]
        points_bottom = [x for x in scene_points_centered if x[0] <= min_thr]
        
        points_right_bottom = [x for x in scene_points_centered if (x[2] >= max_thr) and (x[0] <= min_thr)]
        points_right_top = [x for x in scene_points_centered if (x[2] >= max_thr) and (x[0] >= max_thr)]
        points_left_bottom = [x for x in scene_points_centered if (x[2] <= min_thr) and (x[0] <= min_thr)]
        points_left_top = [x for x in scene_points_centered if (x[2] <= min_thr) and (x[0] >= max_thr)]

        wall_anno = {
            'right': 0,
            'left': 0,
            'top': 0,
            'bottom': 0
        }
        wall_corners_anno = {
            'bottom right': 0,
            'top right': 0,
            'bottom left': 0,
            'top left': 0
        }
        
        if len(points_right) > 5:
            wall_anno['right'] = 1
        if len(points_left) > 5:
            wall_anno['left'] = 1
        if len(points_top) > 5:
            wall_anno['top'] = 1
        if len(points_bottom) > 5:
            wall_anno['bottom'] = 1
        if len(points_right_bottom) > 5:
            wall_corners_anno['bottom right'] = 1
        if len(points_right_top) > 5:
            wall_corners_anno['top right'] = 1
        if len(points_left_bottom) > 5:
            wall_corners_anno['bottom left'] = 1
        if len(points_left_top) > 5:
            wall_corners_anno['top left'] = 1

        all_captions = []
        for k in range(4):
            caption = ''
            wall_corners_anno_sum = np.sum(list(wall_corners_anno.values()))
            wall_anno_sum = np.sum(list(wall_anno.values()))
            if wall_corners_anno_sum == 4:
                caption = ''
            elif wall_corners_anno_sum == 3:
                for key in wall_corners_anno:
                    if wall_corners_anno[key] == 0:
                        caption = rotate_key(key, k) + ' cavity'
            elif wall_corners_anno_sum == 2:
                for key in wall_anno:
                    if wall_anno[key] == 0:
                        caption = 'wall on the ' + rotate_key(key, k)
            elif wall_corners_anno_sum == 1:
                for key in wall_corners_anno:
                    if wall_corners_anno[key] == 1:
                        caption = rotate_key(key, k) + ' corner'
            all_captions += [caption]
                
    else:
        all_captions = ['', '', '', '']
    return all_captions


def get_caption(mesh_groups, across_groups, furniture_cats, close_to_wall, cat_to_name, digit_to_str):
    across_groups_explicit = {}
    for across_group in across_groups:
        across_groups_explicit[across_group[0]] = across_group
        across_groups_explicit[across_group[1]] = (across_group[1], across_group[0])
    group_captions = []
    for mesh_group in mesh_groups:
        mesh_count = {}
        for i in mesh_group:
            cat_name = furniture_cats[i]
            if cat_name not in mesh_count:
                mesh_count[cat_name] = 1
            else:
                mesh_count[cat_name] += 1
        num_objects = 0
        for cat_name in mesh_count:
            num_objects += mesh_count[cat_name]
        if num_objects == 1:
            for cat_name in mesh_count:
                caption = f'a{"n" if cat_to_name[cat_name][0] in vowels else ""} {cat_to_name[cat_name]}'
        elif len(mesh_count) == 2 and num_objects == 2:
            objects = list(mesh_count.keys())
            caption = f'a{"n" if cat_to_name[objects[0]][0] in vowels else ""} {cat_to_name[objects[0]]} next to a{"n" if cat_to_name[objects[1]][0] in vowels else ""} {cat_to_name[objects[1]]}'
        else:
            caption = 'a group of '
            for cat_name in mesh_count:
                if mesh_count[cat_name] == 1:
                    caption += f'a{"n" if cat_to_name[cat_name][0] in vowels else ""} {cat_to_name[cat_name]}'
                else:
                    try:
                        caption += f'{digit_to_str[mesh_count[cat_name]]} {cat_to_name[cat_name]}'
                    except KeyError:
                        caption += f'{mesh_count[cat_name]} {cat_to_name[cat_name]}'
                if mesh_count[cat_name] > 1:
                    caption += 's'
                caption += ', '
            caption = caption[:-2]
        group_captions += [caption]
        
    used_groups = []
    final_caption = ''
    for i, group_caption in enumerate(group_captions):
        if i in used_groups:
            continue
        if i in close_to_wall:
            group_caption = group_caption + ' against the wall'
        if i in across_groups_explicit:
            across_group = across_groups_explicit[i]
            neighbor_caption = group_captions[across_group[1]]
            if across_group[1] not in used_groups:
                if 'next to' in neighbor_caption:
                    neighbor_caption_tokens = neighbor_caption.split(' ')
                    final_caption += f'{group_caption} standing across from a{"n" if neighbor_caption_tokens[1][0] in vowels else ""} {neighbor_caption_tokens[1]} that is next to a{"n" if neighbor_caption_tokens[5][0] in vowels else ""} {neighbor_caption_tokens[5]}'
                    if across_group[1] in close_to_wall:
                        final_caption += ' and against the wall'
                else:
                    if across_group[1] in close_to_wall:
                        neighbor_caption = neighbor_caption + ' against the wall'
                    final_caption += f'{group_caption} standing across from {neighbor_caption}'
            else:
                final_caption += group_caption
                used_groups += [i]
            final_caption += '; '
            used_groups += [across_group[0]]
            used_groups += [across_group[1]]
        else:
            final_caption += group_caption
            final_caption += '; '
            used_groups += [i]
    return final_caption

def get_caption_subcat(mesh_groups, across_groups, furniture_cats, close_to_wall, digit_to_str):
    across_groups_explicit = {}
    for across_group in across_groups:
        across_groups_explicit[across_group[0]] = across_group
        across_groups_explicit[across_group[1]] = (across_group[1], across_group[0])
    group_captions = []
    for mesh_group in mesh_groups:
        mesh_count = {}
        for i in mesh_group:
            cat_name = furniture_cats[i]
            if cat_name not in mesh_count:
                mesh_count[cat_name] = 1
            else:
                mesh_count[cat_name] += 1
        num_objects = 0
        for cat_name in mesh_count:
            num_objects += mesh_count[cat_name]
        if num_objects == 1:
            for cat_name in mesh_count:
                caption = f'a{"n" if cat_name[0] in vowels else ""} {cat_name}'
        elif len(mesh_count) == 2 and num_objects == 2:
            objects = list(mesh_count.keys())
            caption = f'a{"n" if objects[0][0] in vowels else ""} {objects[0]} next to a{"n" if objects[1][0] in vowels else ""} {objects[1]}'
        else:
            caption = 'a group of '
            for cat_name in mesh_count:
                if mesh_count[cat_name] == 1:
                    caption += f'a{"n" if cat_name[0] in vowels else ""} {cat_name}'
                else:
                    try:
                        caption += f'{digit_to_str[mesh_count[cat_name]]} {cat_name}'
                    except KeyError:
                        caption += f'{mesh_count[cat_name]} {cat_name}'
                if mesh_count[cat_name] > 1:
                    caption += 's'
                caption += ', '
            caption = caption[:-2]
        group_captions += [caption]
        
    used_groups = []
    final_caption = ''
    for i, group_caption in enumerate(group_captions):
        if i in used_groups:
            continue
        if i in close_to_wall:
            group_caption = group_caption + ' against the wall'
        if i in across_groups_explicit:
            across_group = across_groups_explicit[i]
            neighbor_caption = group_captions[across_group[1]]
            if across_group[1] not in used_groups:
                if 'next to' in neighbor_caption:
                    neighbor_caption_tokens = neighbor_caption.split(' ')
                    try:
                        final_caption += f'{group_caption} standing across from a{"n" if neighbor_caption_tokens[1][0] in vowels else ""} {neighbor_caption_tokens[1]} that is next to a{"n" if neighbor_caption_tokens[5][0] in vowels else ""} {neighbor_caption_tokens[5]}'
                    except:
                        print('neighbor caption:', neighbor_caption)
                        raise ValueError
                    if across_group[1] in close_to_wall:
                        final_caption += ' and against the wall'
                else:
                    if across_group[1] in close_to_wall:
                        neighbor_caption = neighbor_caption + ' against the wall'
                    final_caption += f'{group_caption} standing across from {neighbor_caption}'
            else:
                final_caption += group_caption
                used_groups += [i]
            final_caption += '; '
            used_groups += [across_group[0]]
            used_groups += [across_group[1]]
        else:
            final_caption += group_caption
            final_caption += '; '
            used_groups += [i]
    return final_caption

digit_to_str = {
    1: 'a',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
    10: 'ten',
    11: 'eleven',
    12: 'twelve',
    13: 'thirteen',
    14: 'fourteen',
    15: 'fiveteen',
    16: 'sixteen',
    17: 'seventeen',
    18: 'eightteen',
    19: 'nineteen',
    20: 'twenty'
}


def generate_captions(meta_data, scene_id, chunk_jsonfile, all_moved_furniture_objs, furniture_cats, furniture_subcats, wall_points, scene_points):

    caption_dict = {'Bed': 0,
                    'Pier/Stool': 0,
                    'Cabinet/Shelf/Desk': 0,
                    'Lighting': 0,
                    'Sofa': 0,
                    'Chair': 0,
                    'Table': 0,
                    'Others': 0}
    cat_to_name = {'Bed': 'bed',
                'Pier/Stool': 'stool',
                'Cabinet/Shelf/Desk': 'cabinet',
                'Lighting': 'lighting',
                'Sofa': 'sofa',
                'Chair': 'chair',
                'Table': 'table',
                'Others': 'object'}

    chunk_bounds = meta_data['subchunk_coords']
    # chunk_bounds = meta_data['cell_bound']
    id_meshes_inside = []
    all_points_inside = []
    used_furniture_cats = []
    used_furniture_subcats = []
    for k, furniture_pts in enumerate(all_moved_furniture_objs):
        inside_flag, points_inside_explicit = check_mesh_inside(furniture_pts, chunk_bounds)
        if len(points_inside_explicit) != 0 and inside_flag:
            all_points_inside += [points_inside_explicit]
            used_furniture_cats += [furniture_cats[k]]
            used_furniture_subcats += [furniture_subcats[k]]
        # if inside_flag:
            caption_dict[furniture_cats[k]] += 1
            id_meshes_inside += [k]
    all_used_points = [all_moved_furniture_objs[k] for k in id_meshes_inside]
    mesh_groups = create_mesh_groups(all_used_points, dist_thr=0.3)
    across_groups = compute_across_groups(mesh_groups, all_used_points)
    wall_points_inside = get_wall_points_inside(wall_points, chunk_bounds)
    scene_points_inside = get_wall_points_inside(scene_points, chunk_bounds)
    close_to_wall = compute_close_to_wall(mesh_groups, all_used_points, wall_points_inside, dist_thr=0.3)
    wall_captions = add_wall_annotation(wall_points_inside, chunk_bounds)
    layout_captions = add_layout_annotation(scene_points_inside, chunk_bounds)
    caption = get_caption(mesh_groups, across_groups, used_furniture_cats, close_to_wall, cat_to_name, digit_to_str)
    caption_subcat = get_caption_subcat(mesh_groups, across_groups, used_furniture_subcats, close_to_wall, digit_to_str)
    if caption.endswith('; '):
        caption = caption[:-2]
    if caption_subcat.endswith('; '):
        caption_subcat = caption_subcat[:-2]
    if caption == '':
        caption = 'Empty room'
    if caption_subcat == '':
        caption_subcat = 'Empty room'

    return caption, caption_subcat, wall_captions, layout_captions, all_points_inside, used_furniture_cats, used_furniture_subcats
    

# the path to the subset of scenes
CHOSEN_SCENES = '/cluster/balar/abokhovkin/data/Front3D/val_scenes_400_600_main.json'

def compute_caption(num_proc=1, proc=0, make_edits=0):

    all_obj_ids = sorted(os.listdir(CHUNKS_DIR))[:]
    all_obj_ids = [x for x in all_obj_ids if not x.endswith('.json') and not x.endswith('.txt')]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

    # with open(os.path.join(CHOSEN_SCENES), 'r') as fin:
    #     meta_data = json.load(fin)
    # all_obj_ids = sorted([scene_id for scene_id in meta_data])
    # all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

    with open('/cluster/daidalos/abokhovkin/Front3D/room_names.txt', 'r') as fin:
        all_possible_rooms = fin.readlines()
        all_possible_rooms = [x[:-1] for x in all_possible_rooms]

    for scene_id in tqdm(all_obj_ids):

        with open(os.path.join(FRONT3DJSON, f'{scene_id}.json'), 'r', encoding="utf-8") as fin:
            scene_metadata = json.load(fin)

        if "scene" not in scene_metadata:
            print(f"There is no scene data in this json file: {scene_id}")
            continue

        FUTURE3D_METADATA = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture/model_info.json'
        with open(FUTURE3D_METADATA, 'r') as fin:
            future3d_metadata_ = json.load(fin)
        future3d_metadata = {}
        for entry in future3d_metadata_:
            future3d_metadata[entry['model_id']] = entry

        all_rooms = {}
        all_room_names = []
        for room in scene_metadata['scene']['room']:
            room_type = room['type']
            all_room_names += [room_type]
            all_rooms[room_type] = []
            for mesh_token in room['children']:
                if 'mesh' in mesh_token['instanceid']:
                    all_rooms[room_type] += [mesh_token['ref']]

        scene_mesh = []
        mesh_faces = []
        wall_points = []
        scene_points = []
        all_mesh_uids = []
        all_room_meshes = {}
        all_room_meshes_edit = {}
        for mesh_data in scene_metadata["mesh"]:
            used_obj_name = mesh_data["type"].strip()
            if used_obj_name == "":
                used_obj_name = "void"
            # extract the vertices from the mesh_data
            vert = [float(ele) for ele in mesh_data["xyz"]]
            # extract the faces from the mesh_data
            faces = mesh_data["faces"]

            try:
                mesh_faces = [[faces[i], faces[i+1], faces[i+2]] for i in range(len(faces)-2)]
                mesh_faces = np.array(mesh_faces)
                mesh_faces = np.vstack([mesh_faces, mesh_faces[:, ::-1]])
            except:
                continue

            num_vertices = int(len(vert) / 3)
            vertices = np.reshape(np.array(vert), [num_vertices, 3])
            faces = np.reshape(np.array(faces), [-1, 3])
            faces = np.vstack([faces, faces[:, ::-1]])

            mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
            if 'Wall' in used_obj_name:
                wall_points += [trimesh.sample.sample_surface(mesh, int(50 * mesh.area))[0]]
            scene_points += [trimesh.sample.sample_surface(mesh, int(50 * mesh.area))[0]]
            scene_mesh += [mesh]

            mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
            all_mesh_uids += [mesh_data['uid']]
            num_edited_rooms = 0
            for room_name in all_rooms:
                for room_uid in all_rooms[room_name]:
                    if room_uid == mesh_data['uid']:
                        if 'Floor' in used_obj_name:
                            if room_name not in all_room_meshes:
                                all_room_meshes[room_name] = []
                                if np.random.uniform() < 0.33 and num_edited_rooms < 2:
                                    room_name_alt = np.random.choice(all_possible_rooms, 1)[0]
                                    num_edited_rooms += 1
                                else:
                                    room_name_alt = room_name
                                all_room_meshes_edit[room_name_alt] = []
                            all_room_meshes[room_name] += [mesh]
                            all_room_meshes_edit[room_name_alt] += [mesh]

        for room_name in all_room_meshes:
            if len(all_room_meshes[room_name]) != 0:
                all_room_meshes[room_name] = trimesh.util.concatenate(all_room_meshes[room_name])
                all_room_meshes[room_name] = trimesh.sample.sample_surface(all_room_meshes[room_name], 1000)[0]

        for room_name in all_room_meshes_edit:
            if len(all_room_meshes_edit[room_name]) != 0:
                all_room_meshes_edit[room_name] = trimesh.util.concatenate(all_room_meshes_edit[room_name])
                all_room_meshes_edit[room_name] = trimesh.sample.sample_surface(all_room_meshes_edit[room_name], 1000)[0]

        scene_mesh = trimesh.util.concatenate(scene_mesh)
        if len(wall_points) != 0:
            wall_points = np.vstack(wall_points)
        if len(scene_points) != 0:
            scene_points = np.vstack(scene_points)

        # collect all loaded furniture objects
        all_furniture_objs = []
        all_furniture_uids = []
        all_furniture_cats = []
        all_furniture_subcats = []
        # for each furniture element
        for ele in scene_metadata["furniture"]:
            # create the paths based on the "jid"
            folder_path = os.path.join(FUTURE3DMODEL, ele["jid"])
            obj_file = os.path.join(folder_path, "raw_model.obj")
            # if the object exists load it -> a lot of object do not exist
            # we are unsure why this is -> we assume that not all objects have been made public
            if os.path.exists(obj_file) and not "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                # load all objects from this .obj file
                furniture_mesh = trimesh.load(obj_file, force='mesh')

                # extract the name, which serves as category id
                used_obj_name = ""
                if "category" in ele:
                    used_obj_name = ele["category"]
                elif "title" in ele:
                    used_obj_name = ele["title"]
                    if "/" in used_obj_name:
                        used_obj_name = used_obj_name.split("/")[0]
                if used_obj_name == "":
                    used_obj_name = "others"

                all_furniture_objs += [furniture_mesh]
                all_furniture_uids += [ele['uid']]
                if ele["jid"] in future3d_metadata:
                    all_furniture_cats += [future3d_metadata[ele["jid"]]['super-category']]
                    subcat_name = future3d_metadata[ele["jid"]]['category']
                    if subcat_name is None:
                        subcat_name = 'object'
                    if '/' in subcat_name:
                        subcat_name = random.choice(subcat_name.split('/'))
                    subcat_name = subcat_name.lower()
                    subcat_name = subcat_name.strip()
                    all_furniture_subcats += [subcat_name]
                else:
                    all_furniture_cats += ['Unknown']
                    all_furniture_subcats += ['Unknown']

        all_moved_furniture_objs = []
        all_moved_furniture_meshes = []
        furniture_cats = []
        furniture_subcats = []
        # for each room
        for room_id, room in enumerate(scene_metadata["scene"]["room"]):
            # for each object in that room
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    for k, obj in enumerate(all_furniture_objs):
                        if all_furniture_uids[k] == child["ref"]:
                            current_obj = obj.copy()

                            rotation_matrix = Rotation.from_quat(child["rot"])
                            rotation_matrix = rotation_matrix.as_matrix()
                            transform_matrix = np.eye(4)
                            transform_matrix[:3, :3] = rotation_matrix
                            transform_matrix[0, 0] *= child["scale"][0]
                            transform_matrix[1, 1] *= child["scale"][1]
                            transform_matrix[2, 2] *= child["scale"][2]
                            transform_matrix[:3, 3] = child["pos"]

                            vertices = np.array(current_obj.vertices)
                            vertices = np.hstack([vertices, np.ones([len(vertices), 1])])

                            vertices = vertices @ transform_matrix.T
                            current_obj.vertices = vertices[:, :3]

                            all_moved_furniture_objs += [trimesh.sample.sample_surface(current_obj, 500)[0]]
                            all_moved_furniture_meshes += [current_obj]
                            furniture_cats += [all_furniture_cats[k]]
                            furniture_subcats += [all_furniture_subcats[k]]
        all_moved_furniture_meshes = trimesh.util.concatenate(all_moved_furniture_meshes)


        if make_edits:
            all_moved_furniture_objs_ = deepcopy(all_moved_furniture_objs)
            furniture_cats_ = deepcopy(furniture_cats)
            furniture_subcats_ = deepcopy(furniture_subcats)

            num_edits = np.random.randint(1, max(len(all_moved_furniture_objs) // 3, 2))
            passed_edits = 0
            for k_edit in range(num_edits):
                if passed_edits > 3:
                    break
                edit_mode = np.random.choice(['deletion', 'replacement', 'addition'])
                edit_obj_id = np.random.choice(np.arange(len(all_moved_furniture_objs_)))
                if edit_mode == 'deletion':
                    del all_moved_furniture_objs_[edit_obj_id]
                    del furniture_cats_[edit_obj_id]
                    del furniture_subcats_[edit_obj_id]
                elif edit_mode == 'replacement':
                    edit_obj_id_2 = np.random.choice(np.arange(len(all_moved_furniture_objs)))
                    all_moved_furniture_objs_[edit_obj_id] = all_moved_furniture_objs[edit_obj_id_2]
                    furniture_cats_[edit_obj_id] = furniture_cats[edit_obj_id_2]
                    furniture_subcats_[edit_obj_id] = furniture_subcats[edit_obj_id_2]
                elif edit_mode == 'addition':
                    edit_obj_id_2 = np.random.choice(np.arange(len(all_moved_furniture_objs)))
                    all_moved_furniture_objs_ += [all_moved_furniture_objs[edit_obj_id_2]]
                    furniture_cats_ += [furniture_cats[edit_obj_id_2]]
                    furniture_subcats_ += [furniture_subcats[edit_obj_id_2]]
                passed_edits += 1


        chunk_jsonfiles = [x for x in os.listdir(os.path.join(CHUNKS_DIR, scene_id)) if x.endswith('.json')]
        for chunk_jsonfile in chunk_jsonfiles:
            try:
                with open(os.path.join(CHUNKS_DIR, scene_id, chunk_jsonfile), 'r') as fin:
                    meta_data = json.load(fin)
            except:
                print(scene_id, chunk_jsonfile)
                continue

            caption, caption_subcat, wall_captions, layout_captions, \
            all_points_inside, used_furniture_cats, used_furniture_subcats = generate_captions(meta_data, scene_id, chunk_jsonfile, 
                                                                                        all_moved_furniture_objs, furniture_cats, furniture_subcats, 
                                                                                        wall_points, scene_points)
            
            meta_data['caption_spatial'] = caption
            meta_data['caption_subcat_spatial'] = caption_subcat
            meta_data['caption_wall'] = wall_captions
            meta_data['caption_layout'] = layout_captions

            if make_edits:
                caption_, caption_subcat_, _, _, _, _, _ = generate_captions(meta_data, scene_id, chunk_jsonfile, 
                                                                    all_moved_furniture_objs_, furniture_cats_, furniture_subcats_,
                                                                    wall_points, scene_points)

                if 'caption_spatial_edit' in meta_data:
                    del meta_data['caption_spatial_edit']
                    del meta_data['caption_subcat_spatial_edit']

                if caption != caption_:
                    print(caption)
                    print(caption_)
                    meta_data['caption_spatial_edit'] = caption_
                    meta_data['caption_subcat_spatial_edit'] = caption_subcat_

            chunk_bounds = meta_data['subchunk_coords']
            # chunk_bounds = meta_data['cell_bound']

            caption_rooms = []
            caption_room_ids = []
            max_room_name = ''
            max_inside_ratio = 0.0
            for room_idx, room_name in enumerate(all_room_meshes):
                inside_ratio = check_mesh_inside_thr(all_room_meshes[room_name], chunk_bounds)
                if inside_ratio > max_inside_ratio:
                    max_inside_ratio = inside_ratio
                    max_room_name = room_name
                    max_room_name = ' '.join(re.findall('[A-Z][^A-Z]*', max_room_name)).lower()
                if inside_ratio >= 0.25:
                    room_name_split = ' '.join(re.findall('[A-Z][^A-Z]*', room_name))
                    caption_rooms += [room_name_split.lower()]
                    caption_room_ids += [room_idx]
            if max_room_name not in caption_rooms:
                room_name_split = ' '.join(re.findall('[A-Z][^A-Z]*', max_room_name))
                caption_rooms += [room_name_split.lower()]
            meta_data['caption_room'] = caption_rooms

            caption_rooms_with_objects = {}
            caption_rooms_with_subcat_objects = {}
            for room_idx, room_name in enumerate(all_room_meshes):
                if room_idx not in caption_room_ids:
                    continue
                caption_rooms_with_objects[room_name] = []
                caption_rooms_with_subcat_objects[room_name] = []
                room_points = all_room_meshes[room_name]
                room_points_xy = room_points[:, [0, 2]]
                tree = cKDTree(room_points_xy)
                for k_obj, object_points in enumerate(all_points_inside):
                    min_x_coord = object_points[:, 0].min()
                    max_x_coord = object_points[:, 0].max()
                    min_y_coord = object_points[:, 2].min()
                    max_y_coord = object_points[:, 2].max()
                    p_1 = np.array([min_x_coord, min_y_coord])
                    p_2 = np.array([max_x_coord, min_y_coord])
                    p_3 = np.array([min_x_coord, max_y_coord])
                    p_4 = np.array([max_x_coord, max_y_coord])
                    bound_points = np.stack([p_1, p_2, p_3, p_4]) # [4, 2]

                    dists, idx = tree.query(bound_points)
                    num_bound_points_inside = np.sum(dists < 0.12)

                    if num_bound_points_inside >= 2:
                        caption_rooms_with_objects[room_name] += [used_furniture_cats[k_obj]]
                        caption_rooms_with_subcat_objects[room_name] += [used_furniture_subcats[k_obj]]

            meta_data['caption_room_cats'] = caption_rooms_with_objects
            meta_data['caption_rooms_subcats'] = caption_rooms_with_subcat_objects

            if make_edits:
                caption_rooms_edit = []
                max_room_name = ''
                max_inside_ratio = 0.0
                for room_name in all_room_meshes_edit:
                    inside_ratio = check_mesh_inside_thr(all_room_meshes_edit[room_name], chunk_bounds)
                    if inside_ratio > max_inside_ratio:
                        max_inside_ratio = inside_ratio
                        max_room_name = room_name
                        max_room_name = ' '.join(re.findall('[A-Z][^A-Z]*', max_room_name)).lower()
                    if inside_ratio >= 0.25:
                        room_name_split = ' '.join(re.findall('[A-Z][^A-Z]*', room_name))
                        caption_rooms_edit += [room_name_split.lower()]
                if max_room_name not in caption_rooms_edit:
                    room_name_split = ' '.join(re.findall('[A-Z][^A-Z]*', max_room_name))
                    caption_rooms_edit += [room_name_split.lower()]
                if caption_rooms != caption_rooms_edit:
                    meta_data['caption_room_edit'] = caption_rooms_edit

            with open(os.path.join(CHUNKS_DIR, scene_id, chunk_jsonfile), 'w') as fout:
                json.dump(meta_data, fout)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    parser.add_argument('-e', '--make_edits', default=0, type=int)
    args = parser.parse_args()

    compute_caption(args.num_proc, args.proc, args.make_edits)