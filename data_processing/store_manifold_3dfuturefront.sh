#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --partition=submit

export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate scenefactor

/rhome/abokhovkin/miniconda3/envs/scenefactor/bin/python -u store_manifold_3dfuturefront.py