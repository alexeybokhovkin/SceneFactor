import os, sys
import numpy as np
import json
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from copy import deepcopy


FRONT3DJSON = '/cluster/himring/pkocsis/datasets/3D-Front/3D-FRONT'
FUTURE3DMODEL = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'
FUTURE3D_METADATA = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture/model_info.json'

SAVE_DIR = '/cluster/falas/abokhovkin/data/Front3D/manifold_meshes'


def compute_scenes(num_proc=1, proc=0):

    all_obj_ids = sorted(os.listdir(FRONT3DJSON))[:]
    all_obj_ids = [x for i, x in enumerate(all_obj_ids) if i % num_proc == proc]

    with open(FUTURE3D_METADATA, 'r') as fin:
        future3d_metadata_ = json.load(fin)
    future3d_metadata = {}
    for entry in future3d_metadata_:
        future3d_metadata[entry['model_id']] = entry


    for jsonfile in tqdm(all_obj_ids):
        scene_name = jsonfile.split('.')[0]
        LOCAL_SAVE_DIR = os.path.join(SAVE_DIR, scene_name)
        # if os.path.exists(LOCAL_SAVE_DIR):
        #     continue
        # if os.path.exists(os.path.join(LOCAL_SAVE_DIR, 'furniture_points_canonic.json')):
        #     continue
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
        
        with open(os.path.join(FRONT3DJSON, jsonfile), 'r', encoding="utf-8") as fin:
            scene_metadata = json.load(fin)
            
        if "scene" not in scene_metadata:
            print(f"There is no scene data in this json file: {jsonfile}")
            continue
            
        scene_mesh = []
        cabinet_mesh = []
        for mesh_data in scene_metadata["mesh"]:
            used_obj_name = mesh_data["type"].strip()
            if used_obj_name == "":
                used_obj_name = "void"
            # extract the vertices from the mesh_data
            vert = [float(ele) for ele in mesh_data["xyz"]]
            # extract the faces from the mesh_data
            faces = mesh_data["faces"]
            
            num_vertices = int(len(vert) / 3)
            vertices = np.reshape(np.array(vert), [num_vertices, 3])
            faces = np.reshape(np.array(faces), [-1, 3])
            faces = np.vstack([faces, faces[:, ::-1]])

            mesh = trimesh.base.Trimesh(vertices=vertices, faces=faces)
            if 'Cabinet' in mesh_data['type']:
                cabinet_mesh += [mesh]
            else:
                if 'Ceiling' not in mesh_data['type'] and 'Top' not in mesh_data['type']:
                    scene_mesh += [mesh]
            
        scene_mesh = trimesh.util.concatenate(scene_mesh)
        if len(cabinet_mesh) != 0:
            cabinet_mesh = trimesh.util.concatenate(cabinet_mesh)
        
        
        # collect all loaded furniture objects
        all_furniture_objs = []
        all_furniture_uids = []
        all_furniture_cats = []
        all_furniture_points = []
        all_furniture_points_canonic = []
        # for each furniture element
        for ele in scene_metadata["furniture"]:
            # create the paths based on the "jid"
            folder_path = os.path.join(FUTURE3DMODEL, ele["jid"])
            if os.path.exists(os.path.join(folder_path, "raw_model_fixed.obj")):
                obj_file = os.path.join(folder_path, "raw_model_fixed.obj")
            else:
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

                surface_points, _ = trimesh.sample.sample_surface(furniture_mesh, int(400 * furniture_mesh.area))
                if len(surface_points) == 0:
                    surface_points, _ = trimesh.sample.sample_surface(furniture_mesh, 10)
                surface_points_canonic = deepcopy(surface_points)
                surface_points_canonic = surface_points_canonic - surface_points_canonic.mean(axis=0)
                max_coord = surface_points_canonic.max()
                surface_points_canonic = surface_points_canonic / (2 * max_coord) / 1.1
                surface_points_canonic = surface_points_canonic + [0.5, 0.5, 0.5]

                all_furniture_points += [surface_points]
                all_furniture_points_canonic += [surface_points_canonic]
                    
                all_furniture_objs += [furniture_mesh]
                all_furniture_uids += [ele['uid']]
                if ele["jid"] in future3d_metadata:
                    all_furniture_cats += [future3d_metadata[ele["jid"]]['super-category']]
                    print(1)
                else:
                    all_furniture_cats += ['Unknown']
                    print(0)
                
        all_moved_furniture_objs = []
        all_moved_furniture_points = []
        all_moved_furniture_points_canonic = []
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

                            current_surface_points = deepcopy(all_furniture_points[k])
                            current_surface_points_canonic = deepcopy(all_furniture_points_canonic[k])
                            current_surface_points = np.array(current_surface_points)
                            current_surface_points = np.hstack([current_surface_points, np.ones([len(current_surface_points), 1])])
                            current_surface_points = current_surface_points @ transform_matrix.T
                            current_surface_points = current_surface_points[:, :3]
                            all_moved_furniture_points += [current_surface_points]
                            all_moved_furniture_points_canonic += [current_surface_points_canonic]
                            
                            all_moved_furniture_objs += [current_obj]
                            furniture_cats += [all_furniture_cats[k]]
        furniture_merged = trimesh.util.concatenate(all_moved_furniture_objs)

        if isinstance(cabinet_mesh, trimesh.base.Trimesh):
            all_moved_furniture_objs += [cabinet_mesh]
            furniture_cats += ['Cabinet/Shelf/Desk']

            surface_points, _ = trimesh.sample.sample_surface(cabinet_mesh, int(400 * cabinet_mesh.area))
            if len(surface_points) == 0:
                surface_points, _ = trimesh.sample.sample_surface(furniture_mesh, 10)
            surface_points_canonic = deepcopy(surface_points)
            surface_points_canonic = surface_points_canonic - surface_points_canonic.mean(axis=0)
            max_coord = surface_points_canonic.max()
            surface_points_canonic = surface_points_canonic / (2 * max_coord) / 1.1
            surface_points_canonic = surface_points_canonic + [0.5, 0.5, 0.5]
            all_moved_furniture_points += [surface_points]
            all_moved_furniture_points_canonic += [surface_points_canonic]

        all_furniture_points = {}
        for obj_id, furniture_obj in enumerate(all_moved_furniture_objs):
            try:
                surface_points, _ = trimesh.sample.sample_surface(furniture_obj, int(200 * furniture_obj.area))
                all_furniture_points[str(obj_id)] = [surface_points.tolist(), furniture_cats[obj_id]]
            except:
                continue
        try:
            surface_points, _ = trimesh.sample.sample_surface(scene_mesh, int(100 * scene_mesh.area))
        except:
            continue
        all_scene_points = {'0': surface_points.tolist()}

        with open(os.path.join(LOCAL_SAVE_DIR, 'furniture_points.json'), 'w') as fin:
            json.dump(all_furniture_points, fin)
        with open(os.path.join(LOCAL_SAVE_DIR, 'scene_points.json'), 'w') as fin:
            json.dump(all_scene_points, fin)
        
        scene_mesh.export(os.path.join(LOCAL_SAVE_DIR, 'scene.obj'))
        for obj_id, furniture_obj in enumerate(all_moved_furniture_objs):
            furniture_obj.export(os.path.join(LOCAL_SAVE_DIR, f'{obj_id}_furniture.obj'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    args = parser.parse_args()

    compute_scenes(args.num_proc, args.proc)