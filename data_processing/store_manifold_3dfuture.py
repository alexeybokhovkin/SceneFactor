import os, sys
from tqdm import tqdm


FUTURE3DMESHES = '/cluster/falas/abokhovkin/data/Front3D/meshes'
SAVEDIR = '/cluster/falas/abokhovkin/data/Front3D/manifold_3dfuture'

# Choose Manifold or ManifoldPlus version (Manifold is preferred)
manifold = '/rhome/abokhovkin/projects/Manifold/build'
# manifold = '/home/bohovkin/cluster/abokhovkin_home/projects/ManifoldPlus/build/manifold'

for obj_id in tqdm(os.listdir(FUTURE3DMESHES)):
    LOCAL_SAVEDIR = os.path.join(SAVEDIR, obj_id)
    os.makedirs(LOCAL_SAVEDIR, exist_ok=True)

    os.system(f"{manifold} --input {os.path.join(FUTURE3DMESHES, obj_id, 'raw_model.obj')} --output {os.path.join(LOCAL_SAVEDIR, 'raw_model.obj')} --depth 9")

    break