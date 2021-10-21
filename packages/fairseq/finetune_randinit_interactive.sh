#!/bin/bash
​
#SBATCH --account=def-msilfver
#SBATCH --time 03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --mail-user=fsamir8@mail.ubc.ca
#SBATCH --job-name=finetune_randinit_interactive
​
#define paths
CHECKPT_PREF=/scratch/fsamir8/finetune_randinit/
PREPROCESS=/project/rrg-msilfver/fsamir8/gitksan-data/results
DEVSRC=/project/rrg-msilfver/fsamir8/gitksan-data/data/spreadsheets/seen_unseen_split_w_root_cross_table/fairseq/gitksan-dev.src
OUT_PREF=/scratch/fsamir8/finetune_randinit/intr_results
​
# init environment
cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate
​
fairseq-interactive --path $CHECKPT_PREF/checkpoint_best.pt --beam 5 --source-lang src \
--target-lang tgt $PREPROCESS