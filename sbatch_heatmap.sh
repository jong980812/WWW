#!/bin/bash
#SBATCH --job-name image_heatmap_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=25G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch
#SBATCH -w augi1
#SBATCH -o /data/jongseo/project/WWW/%A-%x.out
#SBATCH -e /data/jongseo/project/WWW/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=1245

# Set the path to save checkpoints
# OUTPUT_DIR='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/eval_lr_5e-4_epoch_100'
# path to UCF101 annotation file (train.csv/val.csv/test.csv)
# DATA_PATH='/local_datasets/ai_hub_sketch_mw/01/val'
# path to pretrain model
# MODEL_PATH='/data/psh68380/repos/VideoMAE/ucf_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_0.75_videos_e3200/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
# y4는 asd 폴더, y6은 ASD 폴더
python -u /data/jongseo/project/WWW/image_heatmap_vit.py \
--example_root /local_datasets/ILSVRC2012/val \
--heatmap_save_root /data/jongseo/project/WWW/heatmap/prev_mlp/new_layer_0 \
--num_example 5 \
--map_root /data/jongseo/project/WWW/heatmap_info/prev_mlp/new_layer_0 \
--shapley_root /data/jongseo/project/WWW/shapley/vit_base_cls_blocks.0.drop_path1_in1K_class_shap.pkl \
--model vit 
    

echo "Job finish"
exit 0