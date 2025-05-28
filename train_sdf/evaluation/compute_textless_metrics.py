import os, sys
import numpy as np
import json
import torch

sys.path.append('/rhome/abokhovkin/projects/Diffusion-SDF/train_sdf/metrics/pytorch_pcd_metrics/pytorch_structural_losses')
import StructuralLosses as stb

GT_PC = '/cluster/balar/abokhovkin/data/Front3D/gt_pc_2.json' # path to GT points
PD_PC = '/cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/visuals/block_separate_full/pd_pc_block.json' # path to predicted points

SAVE_DIR = '/'.join(PD_PC.split('/')[:-1])

np.random.seed(123)
random_indices = np.random.choice(np.arange(4096), 2048, replace=False)

with open(GT_PC, 'r') as fin:
    gt_pc = json.load(fin)['points']
    gt_pc = np.array(gt_pc)
    gt_pc = gt_pc[:, random_indices, :]
    gt_pc = gt_pc[:2048]
    gt_pc = torch.FloatTensor(gt_pc).cuda().contiguous()

with open(PD_PC, 'r') as fin:
    pd_pc = json.load(fin)['points']
    pd_pc = np.array(pd_pc)
    pd_pc = pd_pc[:, random_indices, :]
    pd_pc = pd_pc[:2048]
    pd_pc = torch.FloatTensor(pd_pc).cuda().contiguous()


results = stb.compute_all_metrics(
    pd_pc, gt_pc, batch_size=64)
print(f'CD&EMD-Metrics: {results}')
# write the dict to a yaml file
out_path = os.path.join(f'cd_emd.txt')
results = {k: v.item() for k, v in results.items()}
with open(os.path.join(SAVE_DIR, 'result_pd_pc.json'), 'w') as fout:
    json.dump(results, fout)