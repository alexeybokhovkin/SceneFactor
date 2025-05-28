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
module load cuda/11.3

cd train_sdf

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3 TORCH_DISTRIBUTED_DEBUG=DETAIL /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u test_listener.py -e /cluster/falas/abokhovkin/rgb-d-diffusion/Diffusion-SDF/stage1_sdf_new/listener_model_2/ -r epoch=5
