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
module load cuda/11.7

cd train_sdf

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u inference_separate_geo.py -e /cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage3_uncond_new/geo_2ch_16spatial_lowres_newdec_vqorig_2x2/ -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u inference_separate_geo.py -e /cluster/andram/abokhovkin/SceneFactor/stage3_uncond_new/geo_16_spatial_old_l2_camready -d /cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/test_sem_maps -s /cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/test_sem_maps -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}
