import json
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, AdamW
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import random
from scipy import ndimage
from tqdm.auto import tqdm

from model import * 
from diffusion import * 
from utils.helpers import * 
from openai_model import UNetModel
from context_encoding import ContextEncoder, BERTEmbedder
from Scheduler import GradualWarmupScheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare_dataset(rank, world_size, data_path, split_file, cond_path, total_pc_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = Dataset(data_path, split_file, cond_path, total_pc_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=True, sampler=sampler)
    
    return dataloader

def corrupt_sem_vox(sem_subchunk):
    obj_coords = np.where(sem_subchunk >= 1)
    obj_coords = np.vstack(obj_coords).T
    max_size = np.array(sem_subchunk.shape)

    if len(obj_coords) > 0:
        # sample random obj locations
        num_loc = np.random.randint(low=1, high=10)
        locs_idx = np.random.choice(np.arange(len(obj_coords)), num_loc)
        locs = obj_coords[locs_idx]
        sem_ids = sem_subchunk[locs[:, 0], locs[:, 1], locs[:, 2]]
        
        # sample window halfsizes
        windows_sizes = np.random.randint(low=1, high=5, size=num_loc)
        windows = [[[locs[i][0] - windows_sizes[i], 
                    locs[i][1] - windows_sizes[i], 
                    locs[i][2] - windows_sizes[i]], 
                    [locs[i][0] + windows_sizes[i], 
                    locs[i][1] + windows_sizes[i], 
                    locs[i][2] + windows_sizes[i]]] for i in range(len(locs))]
        windows = [np.array(x) for x in windows]
        windows_cut = [np.array([np.maximum(x[0], 0), np.minimum(x[1], max_size - 1)]) for x in windows]
        
        for i, window in enumerate(windows_cut):
            sem_id = sem_ids[i]
            
            window_coords = []
            for i_x in range(window[0][0], window[1][0] + 1):
                for i_y in range(window[0][1], window[1][1] + 1):
                    for i_z in range(window[0][2], window[1][2] + 1):
                        window_coords += [[i_x, i_y, i_z]]
            window_coords = np.array(window_coords)
            
            if np.random.uniform() < 0.35:
                # random empty space
                prob = np.random.uniform(high=0.5)
                random_idx = [1 if np.random.uniform() < prob else 0 for _ in range(len(window_coords))]
                random_idx = np.where(random_idx)[0]
                random_coords = window_coords[random_idx]
                sem_subchunk[random_coords[:, 0], random_coords[:, 1], random_coords[:, 2]] = 0
            
            if np.random.uniform() < 0.35:
                # random object space
                prob = np.random.uniform(high=0.5)
                random_idx = [1 if np.random.uniform() < prob else 0 for _ in range(len(window_coords))]
                random_idx = np.where(random_idx)[0]
                random_coords = window_coords[random_idx]
                sem_subchunk[random_coords[:, 0], random_coords[:, 1], random_coords[:, 2]] = sem_id
                
            if np.random.uniform() < 0.45:
                # random dilation and erosion
                window_explicit = sem_subchunk[window[0, 0]:window[1, 0],
                                            window[0, 1]:window[1, 1],
                                            window[0, 2]:window[1, 2]]
                window_explicit = np.where(window_explicit == sem_id, 1, 0)
                if 0 in window_explicit:
                    if np.random.uniform() < 0.5:
                        window_explicit = ndimage.binary_dilation(window_explicit).astype(window_explicit.dtype)
                        if np.random.uniform() < 0.25:
                            window_explicit = ndimage.binary_dilation(window_explicit).astype(window_explicit.dtype)
                    else:
                        window_explicit = ndimage.binary_dilation(1 - window_explicit).astype(window_explicit.dtype)
                        window_explicit = 1 - window_explicit
                        if np.random.uniform() < 0.25:
                            window_explicit = ndimage.binary_dilation(1 - window_explicit).astype(window_explicit.dtype)
                            window_explicit = 1 - window_explicit
                    sem_id_coords = np.vstack(np.where(sem_subchunk == sem_id)).T
                    sem_id_coords_filtered = [x for x in sem_id_coords if (x[0] >= window[0][0]) & (x[0] < window[1][0]) & (x[1] >= window[0][1]) & (x[1] < window[1][1]) & (x[2] >= window[0][2]) & (x[2] < window[1][2])]
                    sem_id_coords_filtered = np.array(sem_id_coords_filtered)
                    if len(sem_id_coords_filtered) != 0:
                        sem_subchunk[sem_id_coords_filtered[:, 0], sem_id_coords_filtered[:, 1], sem_id_coords_filtered[:, 2]] = 0
                        window_explicit_coords = np.where(window_explicit)
                        window_explicit_coords = np.vstack(window_explicit_coords).T + window[0]
                        sem_subchunk[window_explicit_coords[:, 0], window_explicit_coords[:, 1], window_explicit_coords[:, 2]] = sem_id
    return sem_subchunk


class Dataset(Dataset):
    def __init__(self, split_file, cond_path=None, cond_mode=None):
        super().__init__()

        self.cond = cond_path is not None
        self.cond_mode = cond_mode

        if not self.cond:
            self.modulations = unconditional_load_modulations(split_file)
        else:
            self.vox_paths = []
            self.modulations = unconditional_load_modulations(split_file)
            for filename in self.modulations:
                tokens = filename.split('/')
                self.vox_paths += [(os.path.join(cond_path, tokens[-2]), tokens[-1])]
                # self.vox_paths += [(os.path.join(cond_path, tokens[-3]), tokens[-2])]

            if self.cond_mode == 'text_qwen':
                split_file_qwen = split_file[:-4] + '_qwen.txt'
                self.qwen_vox_paths = []
                self.qwen_modulations = unconditional_load_modulations_from_file(split_file_qwen, f_name="latent.npy")
                for filename in self.qwen_modulations:
                    tokens = filename.split('/')
                    self.qwen_vox_paths += [(os.path.join(cond_path, tokens[-3]), tokens[-2])]

        print("Dataset length: ", len(self.modulations))
    
        if self.cond:
            assert len(self.vox_paths) == len(self.modulations)
        
        
    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):
        if self.cond_mode == 'text':
            modulations = torch.from_numpy(np.load(self.modulations[index])[0]).float()

            # only for 2x2 (or 3x3) chunks
            c, h, w, d = modulations.shape
            modulations_buf = torch.zeros((1, 16, 16, 16)).float()
            modulations_buf[:, :h, :w, :d] = modulations
            modulations = modulations_buf

            # scene chunk
            vox_path, chunk_id = self.vox_paths[index]
            chunk_tokens = chunk_id.split('_')
            new_chunk_id = '_'.join([chunk_tokens[0], chunk_tokens[1], chunk_tokens[2], chunk_tokens[3]])
            complete_vox_path = os.path.join(vox_path, f'{new_chunk_id}.npy')

            complete_json_path = os.path.join(vox_path, f'{new_chunk_id}.json')
            with open(complete_json_path, 'r') as fin:
                meta_data = json.load(fin)

            captions_layout_fixed = {
                "top left corner": "with walls positioned at the top and left",
                "top right corner": "with walls positioned at the top and right",
                "bottom left corner": "with walls positioned at the bottom and left",
                "bottom right corner": "with walls positioned at the bottom and right",
                "wall on the top": "with walls positioned at the top",
                "wall on the bottom": "with walls positioned at the bottom",
                "wall on the left": "with walls positioned at the left",
                "wall on the right": "with walls positioned at the right",
                "": ""
            }

            try:
                # caption_keys = ['caption_list', 'caption_subcat_list', 'caption_list_explicit', 'caption_subcat_list_explicit', 'caption_spatial', 'caption_subcat_spatial', 'caption_room']
                caption_keys = ['caption_room_qwen', 'caption_room_cats_qwen', 'caption_rooms_subcats_qwen']
                caption_key = random.choice(caption_keys)
                caption_list = meta_data[caption_key]

                caption = caption_list

                caption_walls = meta_data['caption_layout_manual']
            except:
                caption = ''
                caption_walls = ['', '', '', '']

            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            caption_wall = caption_walls[p_rot]
            caption_wall = captions_layout_fixed[caption_wall]

            if caption.endswith('.'):
                caption = caption[:-1]
            if caption_wall != '':
                caption = caption + ', ' + caption_wall
            modulations = torch.FloatTensor(modulations)
            modulations = torch.rot90(modulations, k=p_rot, dims=[1, 3])

            return modulations, caption
        
        elif self.cond_mode == 'text_qwen':
            qwen_sample = np.random.uniform() > 0.5
            
            if not qwen_sample:
                modulations = torch.from_numpy(np.load(self.modulations[index])[0]).float()
            else:
                index = np.random.choice(np.arange(len(self.qwen_modulations)))
                modulations = torch.from_numpy(np.load(self.qwen_modulations[index])[0]).float()

            # only for 2x2 (or 3x3) chunks
            c, h, w, d = modulations.shape
            modulations_buf = torch.zeros((c, 8, 8, 8)).float()
            modulations_buf[:, :h, :w, :d] = modulations
            modulations = modulations_buf

            # scene chunk
            if not qwen_sample:
                vox_path, chunk_id = self.vox_paths[index]
            else:
                vox_path, chunk_id = self.qwen_vox_paths[index]
            chunk_tokens = chunk_id.split('_')
            new_chunk_id = '_'.join([chunk_tokens[0], chunk_tokens[1], chunk_tokens[2], chunk_tokens[3]])
            complete_vox_path = os.path.join(vox_path, f'{new_chunk_id}.npy')

            complete_json_path = os.path.join(vox_path, f'{new_chunk_id}.json')
            with open(complete_json_path, 'r') as fin:
                meta_data = json.load(fin)

            if not qwen_sample:
                try:
                    caption_keys = ['caption_list', 'caption_subcat_list', 'caption_list_explicit', 'caption_subcat_list_explicit', 'caption_spatial', 'caption_subcat_spatial', 'caption_room']
                    caption_key = random.choice(caption_keys)
                    caption_list = meta_data[caption_key]
                    if not isinstance(caption_list, list):
                        if caption_key in ['caption_spatial', 'caption_subcat_spatial']:
                            sep = '; '
                        else:
                            sep = ', '
                        caption_list = caption_list.split(sep)
                    random.shuffle(caption_list)
                    caption = ''
                    for item in caption_list:
                        caption = caption + item + ', '
                    caption = caption[:-2]
                    caption_walls = meta_data['caption_layout']
                except:
                    print(complete_json_path)
                    caption = ''
                    caption_walls = ['', '', '', '']
            else:
                qwen_keys = []
                for key in meta_data:
                    if 'qwen' in key:
                        qwen_keys += [key]
                caption_key = random.choice(qwen_keys)
                caption = meta_data[caption_key]
                caption_walls = meta_data['caption_layout']

            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            caption_wall = caption_walls[p_rot]
            if caption_wall != '':
                caption = caption + ', ' + caption_wall
            modulations = torch.FloatTensor(modulations)
            modulations = torch.rot90(modulations, k=p_rot, dims=[1, 3])

            return modulations, caption
        
        elif self.cond_mode == 'sem':
            modulations = torch.from_numpy(np.load(self.modulations[index])[0]).float()

            # scene chunk
            vox_path, chunk_id = self.vox_paths[index]
            scene_id = vox_path.split('/')[-1]
            chunk_tokens = chunk_id.split('_')
            new_chunk_id = '_'.join([chunk_tokens[0], chunk_tokens[1], chunk_tokens[2], chunk_tokens[3]])

            complete_json_path = os.path.join(vox_path, f'{new_chunk_id}.json')
            with open(complete_json_path, 'r') as fin:
                meta_data = json.load(fin)
            
            semvox_filename = os.path.join(vox_path, f'{new_chunk_id}_semantic.npy')
            sem_vox = np.load(semvox_filename)

            if sem_vox.shape != (32, 16, 32):
                sem_buf = np.zeros((32, 16, 32))
                sem_buf[0:sem_vox.shape[0], 0:sem_vox.shape[1], 0:sem_vox.shape[2]] = sem_vox[:, :, :]
                sem_vox = sem_buf

            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            modulations = torch.FloatTensor(modulations)
            modulations = torch.rot90(modulations, k=p_rot, dims=[1, 3])
            sem_vox = torch.FloatTensor(sem_vox)
            sem_vox = torch.rot90(sem_vox, k=p_rot, dims=[0, 2])
            sem_vox = sem_vox.long()
            onehot_sem = F.one_hot(sem_vox, num_classes=10)
            onehot_sem = torch.permute(onehot_sem, (3, 0, 1, 2)).float()

            # onehot_sem = onehot_sem[:, ::4, ::4, ::4]

            if modulations.shape != (1, 32, 16, 32):
                print('Modulation', self.modulations[index], modulations.shape)
                modulations_buf = torch.zeros((1, 32, 16, 32))
                modulations_buf[:, 0:modulations.shape[1], 0:modulations.shape[2], 0:modulations.shape[3]] = modulations[:, :, :, :]
                modulations = modulations_buf

            return modulations, onehot_sem
        
        elif self.cond_mode == 'refine':
            modulations = torch.from_numpy(np.load(self.modulations[index])[0]).float()
            chunk_size = 32
            b, h, w, d = modulations.shape
            valid_range_min_point = [[0, 0, 0],
                                     [max(0, h // 2 - chunk_size // 2), 1, max(0, d // 2 - chunk_size // 2)]]
            random_min_point = np.random.uniform(low=valid_range_min_point[0], high=valid_range_min_point[1], size=(3,)).astype('int32')
            random_max_point = random_min_point + chunk_size // 2
            modulations_chunk = modulations[:, 
                                            random_min_point[0] * 2:random_max_point[0] * 2,
                                            random_min_point[1] * 2:random_max_point[1] * 2,
                                            random_min_point[2] * 2:random_max_point[2] * 2]

            # scene chunk
            vox_path, chunk_id = self.vox_paths[index]
            
            lat_filename = os.path.join(vox_path, f'{chunk_id}')
            lat_vox = np.load(lat_filename)[0]
            lat_vox_chunk = lat_vox[:,
                                    random_min_point[0]: random_max_point[0],
                                    random_min_point[1]: random_max_point[1],
                                    random_min_point[2]: random_max_point[2]]

            if lat_vox_chunk.shape != (1, 16, 16, 16):
                lat_buf = np.zeros((1, 16, 16, 16))
                lat_buf[:, 0:lat_vox_chunk.shape[1], 0:lat_vox_chunk.shape[2], 0:lat_vox_chunk.shape[3]] = lat_vox_chunk[:, :, :, :]
                lat_vox_chunk = lat_buf

            p_rot = torch.randint(low=0, high=4, size=(1,))[0]
            modulations_chunk = torch.FloatTensor(modulations_chunk)
            modulations_chunk = torch.rot90(modulations_chunk, k=p_rot, dims=[1, 3])
            lat_vox_chunk = torch.FloatTensor(lat_vox_chunk)
            lat_vox_chunk = torch.rot90(lat_vox_chunk, k=p_rot, dims=[1, 3])

            if modulations_chunk.shape != (1, 32, 32, 32):
                print('Modulation', self.modulations[index], modulations_chunk.shape)
                modulations_buf = torch.zeros((1, 32, 32, 32))
                modulations_buf[:, 0:modulations_chunk.shape[1], 0:modulations_chunk.shape[2], 0:modulations_chunk.shape[3]] = modulations[:, :, :, :]
                modulations_chunk = modulations_buf

            return modulations_chunk, lat_vox_chunk
    

def process_tokens(tokens, target_len=8):
    if len(tokens) > target_len - 1:
        tokens = tokens[:target_len - 1]
    tokens += ['.']
    while len(tokens) < target_len:
        tokens += ['']
    return tokens


class Trainer(object):
    def __init__(
        self,
        train_split_file, val_split_file, diffusion_specs, model_specs, 
        train_lr=1e-5, training_iters=100000, desc='',
        save_and_sample_every=10000, save_model=True, print_freq=1000,
        cond_path=None, rank=None, world_size=None, exp_dir=None, 
        resume=None, batch_size=None, workers=None, cond_mode=None,
        **kwargs
    ):
        super().__init__()

        device = 'cuda'
        self.rank = rank
        self.world_size = world_size

        print("description: ", desc)
        self.model = GaussianDiffusion(
                        model=UNetModel(**model_specs).to(device),
                        **diffusion_specs
                        ).to(rank)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        
        self.has_cond = cond_path is not None 
        self.cond_mode = cond_mode
        self.exp_dir = exp_dir
        

        if self.has_cond:
            if 'text' in cond_mode:
                self.context_encoder = BERTEmbedder(n_embed=1280, n_layer=32, device=f'cuda:{rank}').to(rank)
                self.context_encoder = DDP(self.context_encoder, device_ids=[rank], output_device=rank)
            elif cond_mode == 'sem':
                self.hidden_dims = [16, 32, 64, 128, 128] # 288
                self.context_encoder = ContextEncoder(in_channels=10, hidden_dims=self.hidden_dims).to(rank)
                self.context_encoder = DDP(self.context_encoder, device_ids=[rank], output_device=rank)
            elif cond_mode == 'refine':
                self.hidden_dims = [16, 32, 64, 64, 128]
                self.context_encoder = ContextEncoder(in_channels=1, hidden_dims=self.hidden_dims).to(rank)
                self.context_encoder = DDP(self.context_encoder, device_ids=[rank], output_device=rank)

        # self.ema = ExponentialMovingAverage(self.model.parameters(), 0.995)

        self.save_and_sample_every = save_and_sample_every
        self.save_model = save_model
        self.print_freq = print_freq

        self.batch_size = batch_size
        self.training_iters = training_iters

        # optimizer
        # train_lr = 1e-5
        self.opt = AdamW(self.model.parameters(), lr=train_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=1.0, end_factor=0.05, total_iters=500000)
        if self.has_cond:
            if 'text' in cond_mode:
                self.opt_cond = AdamW(self.context_encoder.parameters(), lr=train_lr)
                self.scheduler_cond = torch.optim.lr_scheduler.StepLR(self.opt_cond, step_size=1, gamma=0.9)
            elif cond_mode == 'sem':
                self.opt_cond = AdamW(self.context_encoder.parameters(), lr=5e-5)
                self.scheduler_cond = torch.optim.lr_scheduler.StepLR(self.opt_cond, step_size=1, gamma=0.9)
            elif cond_mode == 'refine':
                self.opt_cond = AdamW(self.context_encoder.parameters(), lr=2e-5)
                self.scheduler_cond = torch.optim.lr_scheduler.StepLR(self.opt_cond, step_size=1, gamma=0.9)


        cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = self.opt,
                            T_max = 100,
                            eta_min = 0,
                            last_epoch = -1
                        )
        self.warmUpScheduler = GradualWarmupScheduler(
                                optimizer = self.opt,
                                multiplier = 2.5,
                                warm_epoch = 150 // 10,
                                after_scheduler = cosineScheduler
                            ) # don't forget to load last_epc
        
        cosineScheduler_cond = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = self.opt_cond,
                            T_max = 100,
                            eta_min = 0,
                            last_epoch = -1
                        )
        self.warmUpScheduler_cond = GradualWarmupScheduler(
                                optimizer = self.opt,
                                multiplier = 2.5,
                                warm_epoch = 150 // 10,
                                after_scheduler = cosineScheduler_cond
                            ) # don't forget to load last_epc

      
        self.results_folder = os.path.join(exp_dir, "results")
        os.makedirs(self.results_folder, exist_ok=True)
        self.resume = os.path.join(self.results_folder, "{}.pt".format(resume)) if resume is not None else None
        if self.has_cond:
            self.encoder_resume = os.path.join(self.results_folder, "{}_encoder.pt".format(resume)) if resume is not None else None

        # step counter state
        self.step = 0
        self.val_step = 0

        if self.resume:
            self.step, self.model, self.opt, loss = load_model(self.model, self.opt, self.resume)
            # ema_path = self.resume[:-3] + '_ema.pt'
            # self.ema = torch.load(ema_path, map_location=f'cuda:{rank}')
            if self.has_cond:
                _, self.context_encoder, self.opt_cond, loss = load_model(self.context_encoder, self.opt_cond, self.encoder_resume)
            
        # dataset and dataloader
        self.ds = Dataset(train_split_file, cond_path, cond_mode)
        self.val_ds = Dataset(val_split_file, cond_path, cond_mode)
        
        dl = DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=workers, drop_last=True)
        self.dl = cycle(dl)
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=workers, drop_last=True)
        self.val_dl = cycle(val_dl)
        save_code_to_conf(exp_dir)
        self.train()

        cleanup()

    def train(self):

        writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", self.exp_dir))

        with tqdm(initial = self.step, total = self.training_iters) as pbar:


            # diff_100 and 1000 loss refers to the losses when t<100 and 100<t<1000, respectively 
            # typically diff_100 approaches 0 while diff_1000 can still be relatively high
            current_loss = 0
            loss_100, loss_500, loss_1000  = [0], [0], [0]
            while self.step < self.training_iters:
                self.model.train()
                data, condition_raw = next(self.dl)
                data = data.to(self.rank)

                if self.has_cond:
                    if 'text' in self.cond_mode:
                        condition = list(condition_raw)
                        condition = self.context_encoder(condition)
                    elif self.cond_mode == 'sem' or self.cond_mode == 'refine':
                        condition = condition_raw.cuda()
                        condition = self.context_encoder(condition)

                # sample time 
                t = torch.randint(0, self.model.module.num_timesteps, (self.batch_size,), device=self.rank).long()
                condition[np.where(np.random.rand(condition.shape[0]) < 0.12)] = 0
                loss, xt, target, pred, unreduced_loss, loss_dict, x_from_v = self.model.module(data, t, ret_pred_x=True, cond=condition)

                writer.add_scalar("Train loss", loss, self.step)
                writer.add_scalar("Train loss (simple)", loss_dict['loss_simple'], self.step)
                writer.add_scalar("Train loss (vlb)", loss_dict['loss_vlb'], self.step)

                loss.backward()

                pbar.set_description(f'loss: {loss.item():.4f}')

                if self.step % self.print_freq==0:
                    # print(loss_100, loss_500, loss_1000)
                    # print(np.array(loss_100).shape, np.array(loss_500).shape, np.array(loss_1000).shape)
                    print("avg loss at {} iters: {}".format(self.step, current_loss / self.print_freq))
                    print("losses per time at {} iters: {}, {}, {}".format(self.step, np.mean(loss_100), np.mean(loss_500), np.mean(loss_1000)))
                    writer.add_scalar("loss 100", np.mean(loss_100), self.step)
                    writer.add_scalar("loss 500", np.mean(loss_500), self.step)
                    writer.add_scalar("loss 1000", np.mean(loss_1000), self.step)
                    current_loss = 0
                    loss_100, loss_500, loss_1000  = [0], [0], [0]



                    if self.step > 0:
                        # val stage
                        print('Start val stage')
                        val_current_loss = 0
                        val_loss_100, val_loss_500, val_loss_1000  = [0], [0], [0]
                        num_val_iters = 3000
                        for k_val in tqdm(range(num_val_iters)):
                            self.model.eval()
                            data, condition = next(self.val_dl)
                            data = data.to(self.rank)

                            if self.has_cond:
                                if 'text' in self.cond_mode:
                                    condition = list(condition)
                                    condition = self.context_encoder(condition)
                                elif self.cond_mode == 'sem' or self.cond_mode == 'refine':
                                    condition = condition.cuda()
                                    condition = self.context_encoder(condition)

                            # sample time 
                            t = torch.randint(0, self.model.module.num_timesteps, (self.batch_size,), device=self.rank).long()
                            condition[np.where(np.random.rand(condition.shape[0]) < 0.12)] = 0
                            val_loss, xt, target, pred, val_unreduced_loss, val_loss_dict = self.model.module(data, t, ret_pred_x=True, cond=condition)

                            val_current_loss += val_loss.detach().item()
                            val_loss_100.extend([x.mean() for x in val_unreduced_loss[t<100].cpu().numpy()])
                            val_loss_500.extend([x.mean() for x in val_unreduced_loss[t<500].cpu().numpy()])
                            val_loss_1000.extend([x.mean() for x in val_unreduced_loss[t>500].cpu().numpy()])

                            writer.add_scalar("Val loss", val_loss, self.step + k_val)

                            self.val_step += 1

                        print("avg val loss at {} iters: {}".format(self.step, val_current_loss / num_val_iters))
                        print("val losses per time at {} iters: {}, {}, {}".format(self.step, np.mean(val_loss_100), np.mean(val_loss_500), np.mean(val_loss_1000)))
                        writer.add_scalar("val loss 100", np.mean(val_loss_100), self.step)
                        writer.add_scalar("val loss 500", np.mean(val_loss_500), self.step)
                        writer.add_scalar("val loss 1000", np.mean(val_loss_1000), self.step)
                        print('Finish val stage')
                        self.model.train()


                current_loss += loss.detach().item()

                # loss_100.extend(unreduced_loss[t<100].cpu().numpy())
                # loss_500.extend(unreduced_loss[t<500].cpu().numpy())
                # loss_1000.extend(unreduced_loss[t>500].cpu().numpy())

                loss_100.extend([x.mean() for x in unreduced_loss[t<100].cpu().numpy()])
                loss_500.extend([x.mean() for x in unreduced_loss[t<500].cpu().numpy()])
                loss_1000.extend([x.mean() for x in unreduced_loss[t>500].cpu().numpy()])

                self.opt.step()
                self.opt.zero_grad()
                if self.has_cond:
                    self.opt_cond.step()
                    self.opt_cond.zero_grad()
                self.step += 1
                pbar.update(1)
                # self.ema.update(self.model.parameters())

                if self.step % 200 == 0 and self.rank == 0:
                    np.save("{}/{}_cond.npy".format(self.results_folder, self.step), condition_raw.detach().cpu().numpy())
                    np.save("{}/{}_x.npy".format(self.results_folder, self.step), data.detach().cpu().numpy())
                    np.save("{}/{}_xpred.npy".format(self.results_folder, self.step), x_from_v.detach().cpu().numpy())
                    np.save("{}/{}_v.npy".format(self.results_folder, self.step), target.detach().cpu().numpy())
                    np.save("{}/{}_vpred.npy".format(self.results_folder, self.step), pred.detach().cpu().numpy())

                if self.step != 0 and self.step % self.save_and_sample_every == 0 and self.rank == 0:
                    save_model(self.step, self.model, self.opt, loss.detach(), "{}/{}.pt".format(self.results_folder, self.step))
                    if self.has_cond:
                        save_model(self.step, self.context_encoder, self.opt_cond, loss.detach(), "{}/{}_encoder.pt".format(self.results_folder, self.step))

                    # torch.save(self.ema, "{}/{}_ema.pt".format(self.results_folder, self.step))
                    writer.flush()
        writer.close()


def main(rank, world_size, specs, aux_args):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    specs['rank'] = rank
    specs['world_size'] = world_size
    specs['exp_dir'] = aux_args.exp_dir
    specs['resume'] = aux_args.resume
    specs['batch_size'] = aux_args.batch_size
    specs['workers'] = aux_args.workers
    Trainer(**specs)


            
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        required=True,
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default=None,
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--batch_size", "-b",
        default=32, type=int
    )

    arg_parser.add_argument(
        "--workers", "-w",
        default=0, type=int
    )

    aux_args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(aux_args.exp_dir, "specs.json")))

    print('Experiment:', aux_args.exp_dir)

    world_size = 1
    mp.spawn(
        main,
        args=(world_size, specs, aux_args),
        nprocs=world_size
    )

