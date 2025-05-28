#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=256gb
#SBATCH --cpus-per-task=12
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

/rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u compute_inference_chunks.py -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}