#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=200gb
#SBATCH --cpus-per-task=2
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

/rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u store_3dfuturefront.py -n ${SLURM_ARRAY_TASK_COUNT} -p ${SLURM_ARRAY_TASK_ID}