#!/usr/bin/env bash

#PBS -N pastis_train
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l ngpus=1
#PBS -l gputype=T4
#PBS -l walltime=50:00:00
#PBS -o pastis_panoptic.out
#PBS -e pastis_panoptic_err.out

module load cuda/11.3.1
module load gcc/5.4.0-2.26
export CUDA_VISIBLE_DEVICES=0
source /home/nazib/miniconda3/etc/profile.d/conda.sh
conda activate detectron
cd ~/PASTIS_training/panoptic-deeplab/
python tools/train_net.py --cfg configs/pastis_panoptic.yaml TRAIN.IMS_PER_BATCH 1 GPUS '(0, )'