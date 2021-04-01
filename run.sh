#!/bin/bash

# wandb login ebbd83c0708fcda75d8830954d38aec2241b7637

python me_train_qa.py --model spanbi --epochs 2 --context --batch-size 8 --overwrite_output_dir
# python me_train_qa.py --model span --epochs 1 --context --batch-size 8 --overwrite_output_dir
