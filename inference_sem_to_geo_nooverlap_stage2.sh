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

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u inference_sem_to_geo_nooverlap_stage2.py -e /cluster/andram/abokhovkin/SceneFactor/stage3_uncond_new/superres_32_spatial_old_l2_camready -s /cluster/andram/abokhovkin/data/Front3D/baselines/SceneFactor_v2/recon_geo_it375k_qwen_superres_450scenes_camready -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}
