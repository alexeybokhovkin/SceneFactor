import os, sys
import numpy as np
import json
import torch

sys.path.append('/rhome/abokhovkin/projects/Diffusion-SDF/train_sdf/metrics/pytorch_pcd_metrics/pytorch_structural_losses')
import StructuralLosses as stb

GT_PC = '/cluster/balar/abokhovkin/data/Front3D/gt_pc_16k.json' # path to GT points
PD_PC = '/cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/3dfuture_1ch_lowlowres_newdec_onlysem_nll_vqorig_2x2_allcap_l1_final_2/recon_separate_sem/pd_pc_nn_analysis.json' # path to predicted points

SAVE_DIR = '/'.join(PD_PC.split('/')[:-1])

with open(GT_PC, 'r') as fin:
    gt_pc = json.load(fin)['points']
    gt_pc = np.array(gt_pc)
    gt_pc = torch.FloatTensor(gt_pc).cuda().contiguous()
    print('GT', gt_pc.shape)

print('PD_PC folder', PD_PC)
with open(PD_PC, 'r') as fin:
    pd_pc = json.load(fin)['points']
    pd_pc = np.array(pd_pc)
    pd_pc = torch.FloatTensor(pd_pc).cuda().contiguous()
    print('PD', pd_pc.shape)

all_cd, all_emd = stb.pairwise_EMD_CD(
    pd_pc, gt_pc, batch_size=128)

all_cd = all_cd.detach().cpu().numpy()
all_emd = all_emd.detach().cpu().numpy()
np.save(os.path.join(SAVE_DIR, 'all_cd_gt_2048.npy'), all_cd)
np.save(os.path.join(SAVE_DIR, 'all_emd_gt_2048.npy'), all_emd)