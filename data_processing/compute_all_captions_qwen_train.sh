#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=16gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.8

/rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u compute_all_captions_qwen_train.py -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}
# /rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u compute_all_captions_qwen_train.py -n 1 -p 0