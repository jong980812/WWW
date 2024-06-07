#!/bin/bash

#SBATCH --job-name example_selection_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=25G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch_ce_ugrad
#SBATCH -w moana-y4
#SBATCH -o /data/psh68380/repos/WWW/sbatch_log/%A-%x.out
#SBATCH -e /data/psh68380/repos/WWW/sbatch_log/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12345

# Set the path to save checkpoints
# OUTPUT_DIR='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/eval_lr_5e-4_epoch_100'
# path to UCF101 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='/local_datasets/ai_hub_sketch_mw/01/val'
# path to pretrain model
# MODEL_PATH='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
# !! y1, y6은 데이터 경로 ai_hub_sketch_mw 부터, y3은 ai_hub부터
# ! asd 데이터는 y4

python -u /data/psh68380/repos/WWW/example_selection.py \
--data_root "/local_datasets/asd/All_ver2/03/val_3x3cropped" \
--layer "l4" \
--save_root "asd_utils/cropped/exampler" \
--img_save_root "asd_utils/cropped/" \
--num_example 10
    

echo "Job finish"
exit 0