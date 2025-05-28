import os
import json
import re
from tqdm.auto import tqdm
import numpy as np
import warnings
from collections import OrderedDict

import torch
import torch.utils.data 
from torch.nn import functional as F

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *
from models import ContextEncoder, BERTEmbedder
from dataloader.test_loader import Testloader
from dataloader.cap_vox_loader import CapVoxloader


@torch.no_grad()
def test_modulations(args):
    
    # load dataset, dataloader, model checkpoint
    # test_split = specs["TestSplit"]
    # test_split = specs["TrainSplit"]
    test_split = specs["TrainSplit"].split('.')[0] + f'_{args.i_split}.txt'
    factor = specs["Factor"]

    if factor == 'sem':
        default_chunk_size = 64
    else:
        default_chunk_size = 16
    
    test_dataset = CapVoxloader(
        test_split,
        return_filename=True,
        rot_index=specs['rot_index'],
        chunk_size=specs.get("chunk_size", default_chunk_size), # 64/128 for geo, 16 for sem
        big_chunk_size=180,
        sampled_latent=factor,
        orig_chunks_dir=specs.get("orig_chunks_dir", None)
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, num_workers=1, shuffle=False)
    save_latent = True

    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "{}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)

    if factor == 'sem':
        model = CombinedModel3DVQOrig.load_from_checkpoint(resume, specs=specs).cuda().eval() # semantic model
    else:
        model = CombinedModel3DVQOrigGeo.load_from_checkpoint(resume, specs=specs).cuda().eval() # geometric model

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            if idx > 1024:
                break

            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))

            vox, filename, gt_idx, caption = data
            gt_idx = torch.LongTensor(gt_idx.cpu())

            all_cls_names = []
            all_sample_names = []
            for local_filename in filename:
                cls_name = local_filename.split("/")[-2]
                sample_name = local_filename.split("/")[-1].split('.')[0]

                outdir = os.path.join(recon_dir, "{}/{}".format(cls_name, sample_name))
                all_cls_names += [cls_name]
                all_sample_names += [sample_name]
            
            if idx < 16:
                latent, _, _ = model.vae_model.encode(vox.cuda())
                latent_input = latent
                recon = model.vae_model.decode(latent_input)
                for k in range(len(latent)):

                    outdir = os.path.join(recon_dir, "{}".format(all_cls_names[k]))
                    os.makedirs(outdir, exist_ok=True)

                    np.save(os.path.join(outdir, f"{all_sample_names[k]}_{str(specs['rot_index'])}.npy"), recon[k:k+1].cpu().numpy())
                    np.save(os.path.join(outdir, f"{all_sample_names[k]}_{str(specs['rot_index'])}_gt.npy"), vox[k:k+1].cpu().numpy())

            if save_latent:
                try:                                             
                    latent, _, _ = model.vae_model.encode(vox.cuda())
                    for k in range(len(latent)):

                        outdir = os.path.join(latent_dir, "{}".format(all_cls_names[k]))
                        os.makedirs(outdir, exist_ok=True)
                        np.save(os.path.join(outdir, f"{all_sample_names[k]}_{str(specs['rot_index'])}.npy"), latent[k:k+1].cpu().numpy())

                except NotImplementedError as e:
                    print(e)
           

@torch.no_grad()
def test_generation():

    print('Start generation')
    factor = specs["Factor"]
    caption_mapping_path = specs["caption_mapping_path"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Start loading checkpoint')
        if factor == 'sem':
            model = CombinedModel3DVQOrig.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
        else:
            model = CombinedModel3DVQOrigGeo.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
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

        if factor == 'geo':
            hidden_dims = [16, 32, 64, 128, 128] # 288
            context_encoder = ContextEncoder(in_channels=1, hidden_dims=hidden_dims).cuda()
            context_ckpt = torch.load(specs["diffusion_ckpt_path_context"])
        elif factor == 'sem':
            context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device=f'cuda').cuda()
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


        test_split = specs["TrainSplit"]
        test_dataset = Testloader(
            test_split,
            return_filename=True,
            rot_index=None,
            return_type='caption' if factor == 'sem' else 'vox',
            random_caption=True,
            caption_type='qwen',
            chunk_size=64,
            double_chunks=True,
            caption_mapping_path=caption_mapping_path
        )

        torch.manual_seed(2809)
        batch_size = 4
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=batch_size, shuffle=True)
        torch.manual_seed(2809)

        with tqdm(test_dataloader) as pbar:
            for idx, data in enumerate(pbar):
                pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
                output, filename, gt_idx = data 
                gt_idx = torch.LongTensor(gt_idx.cpu())

                if factor == 'sem':
                    caption = output
                    caption = list(caption)
                    if 'Empty room' in caption[0]:
                        continue
                    condition = context_encoder(caption)

                else:
                    vox = output
                    vox = vox.cuda()
                    condition = context_encoder(vox)

                samples, _ = model.diffusion_model.generate_conditional(batch_size, cond=condition)

                if factor == 'sem':
                    samples = samples[:, :, :, :8, :]
                recon = model.vae_model.decode(samples)
                for i in range(recon.shape[0]):
                    filename_tokens = filename[i].split('/')
                    scene_id = filename_tokens[-2]
                    chunk_id = filename_tokens[-1].split('.')[0]

                    np.save(os.path.join(recon_dir, "{}_{}_recon.npy".format(scene_id, chunk_id)), recon[i].detach().cpu().numpy())
                    np.save(os.path.join(recon_dir, f'{i + idx * batch_size}_latent.npy'), samples[i].detach().cpu().numpy())
                    np.save(os.path.join(recon_dir, f'{i + idx * batch_size}_vox.npy'), vox[i].detach().cpu().numpy())
                    
                if idx > 128:
                    break
    
    
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
    arg_parser.add_argument(
        "--rot_index", "-i", default=None, type=int,
        help="rotation index for chunks dataloader",
    )
    arg_parser.add_argument(
        "--i_split", "-s", default=0, type=int,
        help="split for latent dumping",
    )

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    specs['rot_index'] = args.rot_index

    recon_dir = os.path.join(args.exp_dir, f"recon_eval")
    os.makedirs(recon_dir, exist_ok=True)
    
    if specs['training_task'] == 'modulation':
        latent_dir = os.path.join(args.exp_dir, "modulations_2x2")
        os.makedirs(latent_dir, exist_ok=True)
        test_modulations(args)

    elif specs['training_task'] == 'generation':
        test_generation()
        

  
