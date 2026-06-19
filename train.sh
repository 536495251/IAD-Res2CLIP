#!/bin/bash
# Train adapters on MVTec AD and VisA separately.
# Checkpoints are saved to ./checkpoints/{mvtec,visa}/.

# --- Train on MVTec AD ---
python train.py \
    --dataset        mvtec \
    --data_path      ./data/MVTec \
    --save_path      ./checkpoints/mvtec 

# --- Train on VisA ---
python train.py \
    --dataset        visa \
    --data_path      ./data/VisA \
    --save_path      ./checkpoints/visa 
