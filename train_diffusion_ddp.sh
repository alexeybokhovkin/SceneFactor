#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=8
#SBATCH --mem=160gb
##SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,gimli,balrog
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.7

cd train_diffusion

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u train_ddp.py -e /cluster/andram/abokhovkin/SceneFactor/stage2_cond_new/sem_16spatial_l2_med_qwen_fixed_final -r 275000 -b 16 -w 8 # semantic
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u train_ddp.py -e /cluster/andram/abokhovkin/SceneFactor/stage2_cond_new/geo_16_spatial_old_l2_camready -r 75000 -b 8 -w 4 # geometric
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u train_ddp.py -e /cluster/andram/abokhovkin/SceneFactor/stage2_cond_new/superres_32_spatial_old_l2_camready -r 325000 -b 8 -w 4 # refine





