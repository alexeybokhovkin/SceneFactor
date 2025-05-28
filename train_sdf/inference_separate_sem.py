import os
import json
import re
from tqdm.auto import tqdm
import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.utils.data

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *
from models import BERTEmbedder
from train_sdf.dataloader.test_loader import TestLoader


@torch.no_grad()
def test_generation(args):

    print('Start generation')
    caption_mapping_path = specs["caption_mapping_path"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start loading checkpoint')
        model = CombinedModel3DVQOrig.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
        ckpt = torch.load(specs["diffusion_ckpt_path"])

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in ckpt['model_state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = ckpt['model_state_dict']

        model.diffusion_model.load_state_dict(model_dict)
        model = model.cuda().eval()
        print('Finish loading checkpoint')

        context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device=f'cuda').cuda() # 1280 / 512
        context_ckpt = torch.load(specs["diffusion_ckpt_path_context"])

        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in context_ckpt['model_state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = context_ckpt['model_state_dict']

        context_encoder.load_state_dict(model_dict)
        context_encoder = context_encoder.cuda().eval()


    # test_split = specs["TrainSplit"]
    test_split = specs["TestSplit"].split('.')[0] + f'_{args.i_split}.txt'
    test_dataset = TestLoader(
        test_split,
        return_filename=True,
        return_type='caption',
        random_caption=False,
        caption_type='qwen',
        caption_mapping_path=caption_mapping_path
    )
    batch_size = 4
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=batch_size, shuffle=False)

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            caption, filename, gt_idx = data
            gt_idx = torch.LongTensor(gt_idx.cpu())

            caption = list(caption)
            if 'Empty room' in caption[0]:
                continue
            condition = context_encoder(caption)

            samples, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition)

            samples = samples[:, :, :, :8, :]
            recon = model.vae_model.decode(samples)
            for i in range(recon.shape[0]):
                filename_tokens = filename[i].split('/')
                scene_id = filename_tokens[-2]
                chunk_id = filename_tokens[-1].split('.')[0]

                np.save(os.path.join(args.save_dir, "{}_{}_recon.npy".format(scene_id, chunk_id)), recon[i].detach().cpu().numpy())
                
                meta_data = {}
                meta_data['caption'] = caption[i]
                with open(os.path.join(args.save_dir, "{}_{}_recon.json".format(scene_id, chunk_id)), 'w') as fout:
                    json.dump(meta_data, fout)


    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well",
    )
    arg_parser.add_argument(
        "--i_split", "-s", default=0, type=int,
        help="split for latent dumping",
    )
    arg_parser.add_argument(
        "--save_dir", "-d", required=True,
        help="This is a directory to save results",
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    
    os.makedirs(args.save_dir, exist_ok=True)
    test_generation(args)
        

  
