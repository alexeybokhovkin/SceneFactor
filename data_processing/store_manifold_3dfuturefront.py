import os, sys
from tqdm import tqdm


FRONT3DMESHES = '/cluster/falas/abokhovkin/data/Front3D/meshes'
SAVEDIR = '/cluster/falas/abokhovkin/data/Front3D/manifold_meshes'
manifold = '/rhome/abokhovkin/projects/Manifold/build'
# manifold = '/home/bohovkin/cluster/abokhovkin_home/projects/ManifoldPlus/build/manifold'

for obj_id in tqdm(os.listdir(FRONT3DMESHES)):
    LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
    os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

    os.system(f"{manifold} --input {os.path.join(FRONT3DMESHES, obj_id, 'scene.obj')} --output {os.path.join(LOCAL_SAVEDIR, 'scene.obj')} --depth 9")
    os.system(f"{manifold} --input {os.path.join(FRONT3DMESHES, obj_id, 'furniture.obj')} --output {os.path.join(LOCAL_SAVEDIR, 'furniture.obj')} --depth 9")

    break