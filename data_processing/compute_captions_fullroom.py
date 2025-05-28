import os, sys
import numpy as np
import json
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import glob


CHUNKS_DIR = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_rooms'
FRONT3DJSON = '/cluster/gondor/mdahnert/datasets/front3d/3D-FRONT'
FUTURE3DMODEL = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'

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
        return num_points_inside / num_all_points > 0.25, points_inside_explicit
    

def compute_caption(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(CHUNKS_DIR))[:]
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
                else:
                    all_furniture_cats += ['Unknown']

        all_moved_furniture_objs = []
        all_moved_furniture_meshes = []
        furniture_cats = []
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
        all_moved_furniture_meshes = trimesh.util.concatenate(all_moved_furniture_meshes)


        room_dirs = glob.glob(os.path.join(CHUNKS_DIR, scene_id, '*/*'))
        for room_dir in room_dirs:
            chunk_jsonfiles = glob.glob(os.path.join(room_dir, '*.json'))
            room_bounds = None
            for chunk_jsonfile in chunk_jsonfiles:
                if 'room_meta_data' in chunk_jsonfile:
                    continue
                with open(chunk_jsonfile, 'r') as fin:
                    meta_data = json.load(fin)
                if room_bounds is None:
                    room_bounds = meta_data['cell_bound']
                else:
                    room_bounds[0][0] = min(room_bounds[0][0], meta_data['cell_bound'][0][0])
                    room_bounds[0][1] = min(room_bounds[0][1], meta_data['cell_bound'][0][1])
                    room_bounds[0][2] = min(room_bounds[0][2], meta_data['cell_bound'][0][2])
                    room_bounds[1][0] = max(room_bounds[1][0], meta_data['cell_bound'][1][0])
                    room_bounds[1][1] = max(room_bounds[1][1], meta_data['cell_bound'][1][1])
                    room_bounds[1][2] = max(room_bounds[1][2], meta_data['cell_bound'][1][2])
            if room_bounds is None:
                room_bounds = [[0, 0, 0], [1, 1, 1]]

            all_points_inside = []
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
                        'Others': 'other'}
            chunk_bounds = room_bounds
            for k, furniture_pts in enumerate(all_moved_furniture_objs):
                inside_flag, points_inside_explicit = check_mesh_inside(furniture_pts, chunk_bounds)
                if len(points_inside_explicit) != 0:
                    all_points_inside += [points_inside_explicit]
                if inside_flag:
                    caption_dict[furniture_cats[k]] += 1

            caption = ''
            for cat_name in caption_dict:
                if caption_dict[cat_name] > 0:
                    if caption_dict[cat_name] == 1:
                        caption += f'1 {cat_to_name[cat_name]}, '
                    else:
                        caption += f'{caption_dict[cat_name]} {cat_to_name[cat_name]}s, '
            if caption == '':
                caption = 'Empty room'
            else:
                caption = caption[:-2]
            meta_data = {}
            meta_data['full_room_caption'] = caption

            with open(os.path.join(room_dir, 'room_meta_data.json'), 'w') as fout:
                json.dump(meta_data, fout)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_caption(args.num_proc, args.proc)