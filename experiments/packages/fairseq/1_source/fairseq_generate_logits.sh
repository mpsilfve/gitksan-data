#!/bin/bash

#SBATCH --account=def-msilfver
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --mail-user=fsamir@mail.ubc.ca
#SBATCH --job-name=finetune_randinit_interactive_onesource_logit_generation
#SBATCH --output=/scratch/fsamir8/finetune_randinit/1_source/gitksan_interactive.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/1_source/gitksan_interactive.error

#define paths
WORKING_DIR="/project/rrg-msilfver/fsamir8"
ONE_SOURCE_CHECKPT_PREFIX=/scratch/fsamir8/finetune_randinit/1_source
ONE_SOURCE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/1_source
ONE_SOURCE_DEVSRC=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan-dev.src
ONE_SOURCE_TRAIN_SRC=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan-train.src
ONE_SOURCE_SAVEPREF=/scratch/fsamir8/finetune_randinit/1_source

# init environment
cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate

fairseq-interactive --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 10 --source-lang src \
--target-lang tgt $ONE_SOURCE_PREPROC_SAVE_DIR <  $ONE_SOURCE_DEVSRC > $ONE_SOURCE_SAVEPREF/results_logits_train_dev.txt