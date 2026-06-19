#!/bin/bash
# Training-free evaluation (Res2CLIP*) on MVTec AD and VisA.

K_SHOT=1
SAVE_PATH="./results/"

# --- MVTec AD ---
python test.py \
    --mode      training-free \
    --dataset   mvtec \
    --data_path ./data/MVTec \
    --save_path ${SAVE_PATH} \
    --k_shot    ${K_SHOT} \
    --ref_json  ./data/few_shot_records/mvtec_1shot_normal_samples.json

# --- VisA ---
python test.py \
    --mode      training-free \
    --dataset   visa \
    --data_path ./data/VisA \
    --save_path ${SAVE_PATH} \
    --k_shot    ${K_SHOT} \
    --ref_json  ./data/few_shot_records/visa_1shot_normal_samples.json
