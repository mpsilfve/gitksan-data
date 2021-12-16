#!/bin/bash

# Will actually start an fairseq-interactive session. Not meant to be used with `sbatch`.

#define paths
WORKING_DIR="/project/rrg-msilfver/fsamir8"
ONE_SOURCE_CHECKPT_PREFIX=/scratch/fsamir8/finetune_randinit/1_source
ONE_SOURCE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/1_source
ONE_SOURCE_DEVSRC=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan-dev.src
ONE_SOURCE_SEEN_TEST=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan-seen-test.src
ONE_SOURCE_UNSEEN_TEST=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan-unseen-test.src
ONE_SOURCE_SAVEPREF=/scratch/fsamir8/finetune_randinit/1_source

# init environment
cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate

fairseq-interactive --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 10 --source-lang src \
--target-lang tgt $ONE_SOURCE_PREPROC_SAVE_DIR 