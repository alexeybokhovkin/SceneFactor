import os
import json
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.utils.data 
from torch.nn import functional as F

# add paths in model/__init__.py for new models
from models import *
from utils.reconstruct import *
from diff_utils.helpers import *
from dataloader.sdf_vox_loader import SdfVoxLoaderShapeGlotTest


@torch.no_grad()
def evaluate(args):

    # path to GT samples
    gt_dir = '/cluster/balar/abokhovkin/data/Front3D/chunked_data_lowres_inference'
    # path to our model samples
    ours_dir = '/cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/geo_2ch_16spatial_lowres_newdec_vqorig_2x2/recon_semantic_2steps_1_combined' 
    # path to another model samples
    sdfusion_dir = '/cluster/hithlum/qmeng/Experiments/T2SD/baselines/sdfusion/logs_home/text_to_scene'

    test_dataset = SdfVoxLoaderShapeGlotTest(gt_dir,
                                             ours_dir,
                                             sdfusion_dir,
                                             mode='sdfusion',
                                             chunk_size=64,
                                             double_chunks=True,
                                             caption_dir=None, # path to caption the samples were generated with
                                             compare_sets=('gt', 'ours')
                                             )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16, num_workers=8,
            drop_last=True, shuffle=False, pin_memory=True, persistent_workers=True
        )

    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "{}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
    model = Listener.load_from_checkpoint(resume, specs=specs).cuda().eval()

    scores = []
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            voxes = data['gt_voxes'].cuda()
            captions = data['caption']
            filenames = data['filename']

            logits = model(voxes, captions)
            scores.append(F.softmax(logits, dim=1).to('cpu').numpy())

        scores = np.vstack(scores)

    print('Scores', scores.mean(axis=0))
    scores_diff = np.abs(scores[:, 0] - scores[:, 1])
    scores_conf = np.where(scores_diff <= 0.25, 1, 0)
    print('Confusion:', np.sum(scores_conf) / len(scores_diff))


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="continue from previous saved logs, integer value, 'last', or 'finetune'",
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    
    evaluate(args)

