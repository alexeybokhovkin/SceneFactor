#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
##SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,gimli,balrog
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.3

cd train_sdf

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u test.py -e /cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/geo_2ch_16spatial_lowres_newdec_vqorig_2x2/