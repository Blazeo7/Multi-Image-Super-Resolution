#!/bin/bash

source ~/.bashrc
conda activate diarizen
export TORCH_HOME=/mnt/scratch/tmp/xnguye28/torch_cache

cd /pub/users/xnguye28/knn-sr/scripts/homography/
# taskset -c 0 python build_aligned_data.py --input-dir /pub/users/xnguye28/knn-sr/data/raw/synthetic/ --output-dir /pub/users/xnguye28/knn-sr/data/raw/aligned-synthetic/ --scale-factor 2 --skip-existing --ncc-threshold 0.1

export N_GPUS=1
if [ -n "$N_GPUS" ]; then
  export $(/mnt/matylda4/kesiraju/bin/gpus $N_GPUS) || exit 1
  echo "Visible devices: ${CUDA_VISIBLE_DEVICES}"
else
  export CUDA_VISIBLE_DEVICES=""
fi
cd /pub/users/xnguye28/knn-sr/scripts/homography/
taskset -c 0 python build_ds_new.py --input-dir /mnt/matylda1/hradis/2025-03-01_SIFT/data/ --output-dir /pub/users/xnguye28/knn-sr/data/raw/aligned_real --resume
