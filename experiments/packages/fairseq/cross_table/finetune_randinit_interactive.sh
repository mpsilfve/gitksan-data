#!/bin/bash

#SBATCH --account=def-msilfver
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --mail-user=fsamir@mail.ubc.ca
#SBATCH --job-name=finetune_randinit_interactive_cross_table
#SBATCH --output=/scratch/fsamir8/finetune_randinit/cross_table/gitksan_interactive.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/cross_table/gitksan_interactive.error

#define paths
WORKING_DIR="/project/rrg-msilfver/fsamir8"
CROSS_TABLE_CHECKPT_PREFIX=/scratch/fsamir8/finetune_randinit/cross_table
CROSS_TABLE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/cross_table
CROSS_TABLE_DEVSRC=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root_cross_table/fairseq/gitksan-dev.src
CROSS_TABLE_SEEN_TEST=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root_cross_table/fairseq/gitksan-seen-test.src
CROSS_TABLE_UNSEEN_TEST=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root_cross_table/fairseq/gitksan-unseen-test.src
CROSS_TABLE_SAVEPREF=/scratch/fsamir8/finetune_randinit/cross_table

# init environment
cd $WORKING_DIR
source py3env/bin/activate

fairseq-interactive --path $CROSS_TABLE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 4 --source-lang src \
--target-lang tgt $CROSS_TABLE_PREPROC_SAVE_DIR < $CROSS_TABLE_DEVSRC > $CROSS_TABLE_SAVEPREF/results_dev.txt

fairseq-interactive --path $CROSS_TABLE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 4 --source-lang src \
--target-lang tgt $CROSS_TABLE_PREPROC_SAVE_DIR < $CROSS_TABLE_SEEN_TEST > $CROSS_TABLE_SAVEPREF/results_seen_test.txt

fairseq-interactive --path $CROSS_TABLE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 4 --source-lang src \
--target-lang tgt $CROSS_TABLE_PREPROC_SAVE_DIR < $CROSS_TABLE_UNSEEN_TEST > $CROSS_TABLE_SAVEPREF/results_unseen_test.txt