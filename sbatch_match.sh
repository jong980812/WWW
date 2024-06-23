#!/bin/bash

#SBATCH --job-name concept_layer3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 1-00:00:0
#SBATCH --partition batch
#SBATCH -w augi2
#SBATCH -o /data/jongseo/project/WWW/%A-%x.out
#SBATCH -e /data/jongseo/project/WWW/%A-%x.err
echo $PWD
echo $SLURMD_NODENAME
current_time=$(date "+%Y%m%d-%H:%M:%S")

echo $current_time
export MASTER_PORT=12045

# Set the path to save checkpoints
# DATA_PATH='/local_datasets/ai_hub_sketch_mw/01/val'

# batch_size can be adjusted according to number of GPUs
# this script is for 2 GPUs (1 nodes x 2 GPUs)
python -u /data/jongseo/project/WWW/concept_matching.py \
--word_save_root "/data/jongseo/project/WWW/utils/words_feat_80k.pkl" \
--img_save_root "/data/jongseo/project/WWW/example_selection/vit_tiny/vit_layer11_cls" \
--img_feat_root "/data/jongseo/project/WWW/concept_matching/vit_tiny/vit_layer11_cls" \
--concept_sim_root "/data/jongseo/project/WWW/concept_matching/vit_tiny/vit_layer11_cls" \
--concept_root "/data/jongseo/project/WWW/concept_matching/vit_tiny/vit_layer11_cls" \
--layer 11_cls \
--num_example 40 \
--data_size 80
    

echo "Job finish"
exit 0
