#!/bin/bash

#SBATCH --account=def-msilfver
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --mail-user=fsamir@mail.ubc.ca
#SBATCH --job-name=finetune_randinit_gold_scores_onesource
#SBATCH --output=/scratch/fsamir8/finetune_randinit/1_source/gitksan_generate_gold.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/1_source/gitksan_gold.error

WORKING_DIR="/project/rrg-msilfver/fsamir8"
ONE_SOURCE_CHECKPT_PREFIX=/scratch/fsamir8/finetune_randinit/1_source
ONE_SOURCE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/1_source
ONE_SOURCE_SAVEPREF=/scratch/fsamir8/finetune_randinit/1_source

cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate

fairseq-generate $ONE_SOURCE_PREPROC_SAVE_DIR  --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --score-reference --gen-subset train > $ONE_SOURCE_SAVEPREF/gold_scores_train.txt
fairseq-generate $ONE_SOURCE_PREPROC_SAVE_DIR  --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --score-reference --gen-subset valid > $ONE_SOURCE_SAVEPREF/gold_scores_valid.txt
fairseq-generate $ONE_SOURCE_PREPROC_SAVE_DIR  --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --score-reference --gen-subset test   > $ONE_SOURCE_SAVEPREF/gold_scores_test.txt