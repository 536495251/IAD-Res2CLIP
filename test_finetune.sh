#!/bin/bash
# Fine-tuned evaluation (Res2CLIP†) with cross-dataset generalisation:
#   VisA-trained  checkpoint → test on MVTec AD
#   MVTec-trained checkpoint → test on VisA

K_SHOT=1
SAVE_PATH="./results/"
CKPT_MVTEC="./checkpoints/mvtec/checkpoint_ep10.pth"
CKPT_VISA="./checkpoints/visa/checkpoint_ep10.pth"

# --- train on VisA → MVTec AD ---
python test.py \
    --mode       finetune \
    --dataset    mvtec \
    --data_path  ./data/MVTec \
    --save_path  ${SAVE_PATH} \
    --k_shot     ${K_SHOT} \
    --checkpoint ${CKPT_VISA} \
    --res_features_list 6 12 18 24 \
    --ref_json   ./data/few_shot_records/mvtec_1shot_normal_samples.json 

# --- train on MVTec AD → VisA ---
python test.py \
    --mode       finetune \
    --dataset    visa \
    --data_path  ./data/VisA \
    --save_path  ${SAVE_PATH} \
    --k_shot     ${K_SHOT} \
    --checkpoint ${CKPT_MVTEC} \
    --res_features_list 6 12 18 24 \
    --ref_json   ./data/few_shot_records/visa_1shot_normal_samples.json
