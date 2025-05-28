import os
import json
import warnings

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy

# add paths in model/__init__.py for new models
from models import *
from diff_utils.helpers import *
from dataloader.sdf_vox_loader import VoxLoaderSem, VoxLoaderGeo


def train():
    
    # initialize dataset and loader
    split = specs["TrainSplit"]
    val_split = specs["TestSplit"]
    factor = specs["Factor"]

    if factor == 'sem':
        dataset = VoxLoaderSem
        default_chunk_size = 64
    else:
        dataset = VoxLoaderGeo
        default_chunk_size = 16

    train_dataset = dataset(split, 
                            modulation_path=specs.get("modulation_path", None),
                            chunk_size=specs.get("chunk_size", default_chunk_size),
                            predefined_chunk_bounds=specs.get("predefined_chunk_bounds", False),
                            augment_chunks=specs.get("augment_chunks", True),
                            double_chunks=specs.get("double_chunks", True))
    val_dataset = dataset(val_split, 
                          modulation_path=specs.get("modulation_path", None),
                          chunk_size=specs.get("chunk_size", default_chunk_size),
                          predefined_chunk_bounds=specs.get("predefined_chunk_bounds", False),
                          augment_chunks=specs.get("augment_chunks", True),
                          double_chunks=specs.get("double_chunks", True))
    
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            drop_last=True, shuffle=True, pin_memory=True, persistent_workers=True
        )
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=2, num_workers=1,
            drop_last=True, shuffle=False, pin_memory=True, persistent_workers=True
        )

    # creates a copy of current code / files in the config folder
    save_code_to_conf(args.exp_dir) 
    
    # pytorch lightning callbacks 
    callback = ModelCheckpoint(dirpath=args.exp_dir, filename='{epoch}', save_top_k=-1, save_last=True, every_n_epochs=1)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [callback, lr_monitor]

    if factor == 'sem':
        model = CombinedModel3DVQOrig(specs, args.exp_dir) # sem vq
    else:
        model = CombinedModel3DVQOrigGeo(specs, args.exp_dir) # geo vq

    if args.resume == 'finetune':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False)
            ckpt = torch.load(specs["diffusion_ckpt_path"])
            model.diffusion_model.load_state_dict(ckpt['model_state_dict'])
            context_ckpt = torch.load(specs["diffusion_ckpt_path_context"])
            model.context_encoder.load_state_dict(context_ckpt['model_state_dict'])
        resume = None
    elif args.resume is not None:
        ckpt = "{}.ckpt".format(args.resume)
        resume = os.path.join(args.exp_dir, ckpt)
    else:
        resume = None

    print('CUDA is available:', torch.cuda.is_available())

    trainer = pl.Trainer(devices=-1,
                         accelerator='gpu',
                         precision=32, max_epochs=specs["num_epochs"], callbacks=callbacks, log_every_n_steps=1,
                         default_root_dir=os.path.join("tensorboard_logs", args.exp_dir),
                         num_sanity_val_steps=0,
                         limit_val_batches=400,
                         limit_train_batches=8000,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=resume)

    
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
    arg_parser.add_argument("--batch_size", "-b", default=1, type=int)
    arg_parser.add_argument("--workers", "-w", default=8, type=int)

    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))

    train()
