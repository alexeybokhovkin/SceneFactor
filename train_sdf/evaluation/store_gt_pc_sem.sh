#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate textures

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.3

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/textures/bin/python -u store_gt_pc_sem.py 