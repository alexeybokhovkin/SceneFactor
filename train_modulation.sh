#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
##SBATCH --constraint="rtx_a6000"
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,gimli,balrog
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.7

cd train_sdf

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u train.py -e /cluster/andram/abokhovkin/SceneFactor/stage1_sdf_new/v2_geo_1ch_16spatial_vqorig_lowres -b 32 -w 4


