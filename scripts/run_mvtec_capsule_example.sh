#!/usr/bin/env bash
set -e

# Example run script. Adjust paths!
DATASET_PATH=/data/MVTEC_CAPSULE
RESULTS=./results-capsule

python -u train_net.py   --GPU=0   --DATASET=MVTecCapsule   --RUN_NAME=MVTecCapsule_capsule_mixed   --DATASET_PATH=${DATASET_PATH}   --RESULTS_PATH=${RESULTS}   --SAVE_IMAGES=True   --DILATE=7   --EPOCHS=50   --LEARNING_RATE=1.0   --DELTA_CLS_LOSS=0.01   --BATCH_SIZE=1   --WEIGHTED_SEG_LOSS=True   --WEIGHTED_SEG_LOSS_P=2   --WEIGHTED_SEG_LOSS_MAX=1   --DYN_BALANCED_LOSS=True   --GRADIENT_ADJUSTMENT=True   --FREQUENCY_SAMPLING=True   --TRAIN_NUM=$(wc -l < ${DATASET_PATH}/splits/MVTecCapsule/train.txt)   --NUM_SEGMENTED=$(wc -l < ${DATASET_PATH}/splits/MVTecCapsule/segmented_train.txt)
