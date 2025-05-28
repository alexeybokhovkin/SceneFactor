#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=24gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

/rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u compute_train_chunks_vox_lowres_2x2_sem.py -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}