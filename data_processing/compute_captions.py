import os, sys
import numpy as np
import json
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import random
from copy import deepcopy


# CHUNKS_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference' # inference chunks
CHUNKS_DIR = '/cluster/andram/abokhovkin/data/Front3D/train_chunks_sem_lowres_2x2_sem_canonic' # training chunks

FRONT3DJSON = '/cluster/gondor/mdahnert/datasets/front3d/3D-FRONT'
FUTURE3DMODEL = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'


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

vowels = ['a', 'u', 'i', 'e', 'o', 'y']

def generate_captions(meta_data, scene_id, chunk_jsonfile, all_moved_furniture_objs, furniture_cats, furniture_subcats):
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

    # if 'caption_list' in meta_data and 'caption_subcat_list' in meta_data and 'caption_list_explicit' in meta_data and 'caption_subcat_list_explicit' in meta_data:
    #     return
    caption_subcats = []
    chunk_bounds = meta_data['subchunk_coords']
    # chunk_bounds = meta_data['cell_bound']
    for k, furniture_pts in enumerate(all_moved_furniture_objs):
        inside_flag, points_inside_explicit = check_mesh_inside(furniture_pts, chunk_bounds)
        if inside_flag:
            caption_dict[furniture_cats[k]] += 1
            caption_subcats += [furniture_subcats[k]]

    caption_subcats_dict = {}
    if len(caption_subcats) != 0:
        for subcat in caption_subcats:
            if subcat not in caption_subcats_dict:
                caption_subcats_dict[subcat] = 0
            caption_subcats_dict[subcat] += 1

    caption = ''
    for cat_name in caption_dict:
        if caption_dict[cat_name] > 0:
            if caption_dict[cat_name] == 1:
                caption += f'a{"n" if cat_to_name[cat_name][0] in vowels else ""} {cat_to_name[cat_name]}, '
            else:
                try:
                    caption += f'{digit_to_str[caption_dict[cat_name]]} {cat_to_name[cat_name]}s, '
                except KeyError:
                    caption += f'{caption_dict[cat_name]} {cat_to_name[cat_name]}s, '
    if caption == '':
        caption = 'Empty room'
    else:
        caption = caption[:-2]

    caption_inexact = ''
    for cat_name in caption_dict:
        if caption_dict[cat_name] > 0:
            if caption_dict[cat_name] == 1:
                caption_inexact += f'a{"n" if cat_to_name[cat_name][0] in vowels else ""} {cat_to_name[cat_name]}, '
            else:
                try:
                    if caption_dict[cat_name] <= 2:
                        spec = 'few'
                    elif 3 <= caption_dict[cat_name] <= 4:
                        spec = 'several'
                    else:
                        spec = 'many'
                    caption_inexact += f'{spec} {cat_to_name[cat_name]}s, '
                except KeyError:
                    caption_inexact += f'many {cat_to_name[cat_name]}s, '
    if caption_inexact == '':
        caption_inexact = 'Empty room'
    else:
        caption_inexact = caption_inexact[:-2]

    caption_subcat = ''
    for cat_name in caption_subcats_dict:
        if caption_subcats_dict[cat_name] > 0:
            if caption_subcats_dict[cat_name] == 1:
                caption_subcat += f'a{"n" if cat_name[0] in vowels else ""} {cat_name}, '
            else:
                try:
                    caption_subcat += f'{digit_to_str[caption_subcats_dict[cat_name]]} {cat_name}s, '
                except KeyError:
                    caption_subcat += f'{caption_subcats_dict[cat_name]} {cat_name}s, '
    if caption_subcat == '':
        caption_subcat = 'Empty room'
    else:
        caption_subcat = caption_subcat[:-2]

    caption_subcat_inexact = ''
    for cat_name in caption_subcats_dict:
        if caption_subcats_dict[cat_name] > 0:
            if caption_subcats_dict[cat_name] == 1:
                caption_subcat_inexact += f'a{"n" if cat_name[0] in vowels else ""} {cat_name}, '
            else:
                try:
                    if caption_dict[cat_name] <= 2:
                        spec = 'few'
                    elif 3 <= caption_dict[cat_name] <= 4:
                        spec = 'several'
                    else:
                        spec = 'many'
                    caption_subcat_inexact += f'{spec} {cat_name}s, '
                except KeyError:
                    caption_subcat_inexact += f'many {cat_name}s, '
    if caption_subcat_inexact == '':
        caption_subcat_inexact = 'Empty room'
    else:
        caption_subcat_inexact = caption_subcat_inexact[:-2]

    caption_explicit = []
    for cat_name in caption_dict:
        if caption_dict[cat_name] > 0:
            for k_inst in range(caption_dict[cat_name]):
                caption_explicit += [f'a{"n" if cat_to_name[cat_name][0] in vowels else ""} {cat_to_name[cat_name]}']
    if len(caption_explicit) == 0:
        caption_explicit = ['Empty room']

    caption_subcat_explicit = []
    for cat_name in caption_subcats_dict:
        if caption_subcats_dict[cat_name] > 0:
            for k_inst in range(caption_subcats_dict[cat_name]):
                caption_subcat_explicit += [f'a{"n" if cat_name[0] in vowels else ""} {cat_name}']
    if len(caption_subcat_explicit) == 0:
        caption_subcat_explicit = ['Empty room']

    return caption, caption_subcat, caption_explicit, caption_subcat_explicit, caption_inexact, caption_subcat_inexact

def compute_caption(num_proc=1, proc=0, make_edits=0):

    all_obj_ids = sorted(os.listdir(CHUNKS_DIR))[:]
    all_obj_ids = [x for x in all_obj_ids if not x.endswith('.json') and not x.endswith('.txt')]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

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

        scene_mesh = []
        mesh_xyz = []
        mesh_faces = []
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
            scene_mesh += [mesh]

        scene_mesh = trimesh.util.concatenate(scene_mesh)


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
                    if subcat_name.endswith(' '):
                        subcat_name = subcat_name[:-1]
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

                            all_moved_furniture_objs += [trimesh.sample.sample_surface(current_obj, 2000)[0]]
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
            with open(os.path.join(CHUNKS_DIR, scene_id, chunk_jsonfile), 'r') as fin:
                meta_data = json.load(fin)
            caption, caption_subcat, caption_explicit, caption_subcat_explicit, caption_inexact, caption_subcat_inexact = generate_captions(meta_data, scene_id, chunk_jsonfile, all_moved_furniture_objs, furniture_cats, furniture_subcats)
            meta_data['caption_list'] = caption
            meta_data['caption_subcat_list'] = caption_subcat
            meta_data['caption_list_explicit'] = caption_explicit
            meta_data['caption_subcat_list_explicit'] = caption_subcat_explicit
            meta_data['caption_list_inexact'] = caption_inexact
            meta_data['caption_subcat_list_inexact'] = caption_subcat_inexact

            if make_edits:
                caption_, caption_subcat_, caption_explicit_, caption_subcat_explicit_, caption_inexact_, caption_subcat_inexact_ = generate_captions(meta_data, scene_id, chunk_jsonfile, all_moved_furniture_objs_, furniture_cats_, furniture_subcats_)

                if 'caption_list_edit' in meta_data:
                    del meta_data['caption_list_edit']
                    del meta_data['caption_subcat_list_edit']
                    del meta_data['caption_list_explicit_edit']
                    del meta_data['caption_subcat_list_explicit_edit']

                if caption != caption_:
                    print(caption)
                    print(caption_)
                    meta_data['caption_list_edit'] = caption_
                    meta_data['caption_subcat_list_edit'] = caption_subcat_
                    meta_data['caption_list_explicit_edit'] = caption_explicit_
                    meta_data['caption_subcat_list_explicit_edit'] = caption_subcat_explicit_
                    meta_data['caption_list_inexact_edit'] = caption_inexact_
                    meta_data['caption_subcat_list_inexact_edit'] = caption_subcat_inexact_

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