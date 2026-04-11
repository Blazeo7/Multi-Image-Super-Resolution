#!/bin/bash

source ~/.bashrc
conda activate diarizen
export TORCH_HOME=/mnt/scratch/tmp/xnguye28/torch_cache

cd /pub/users/xnguye28/knn-sr/scripts/homography/
# taskset -c 0 python build_aligned_data.py --input-dir /pub/users/xnguye28/knn-sr/data/raw/synthetic/ --output-dir /pub/users/xnguye28/knn-sr/data/raw/aligned-synthetic/ --scale-factor 2 --skip-existing --ncc-threshold 0.1

taskset -c 0 python build_real_ds.py --input-dir /mnt/matylda1/hradis/2025-03-01_SIFT/data/ --output-dir /pub/users/xnguye28/knn-sr/data/raw/aligned-real --n-supporting 6 --min-supporting 5 --hr-res 1024 --lr-res 512 --ncc-threshold 0.1
