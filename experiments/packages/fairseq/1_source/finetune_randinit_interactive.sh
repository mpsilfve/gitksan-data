#!/bin/bash

#SBATCH --account=def-msilfver
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca
#SBATCH --job-name=finetune_randinit_interactive_onesource
#SBATCH --output=/scratch/fsamir8/finetune_randinit/1_source/gitksan_interactive.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/1_source/gitksan_interactive.error

#define paths
WORKING_DIR="/project/rrg-msilfver/fsamir8"
ONE_SOURCE_CHECKPT_PREFIX=/scratch/fsamir8/finetune_randinit/1_source
ONE_SOURCE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/1_source
# TODO:
ONE_SOURCE_DATA_SAVE_DIR=$WORKING_DIR/gitksan-data/data/spreadsheets/standard_challenge_split/type_split/cross_product_source/fairseq/


ONE_SOURCE_DEVSRC=$ONE_SOURCE_DATA_SAVE_DIR/gitksan-dev.src
ONE_SOURCE_STANDARD_TEST=$ONE_SOURCE_DATA_SAVE_DIR/gitksan-standard_test.src
ONE_SOURCE_CHALLENGE_TEST=$ONE_SOURCE_DATA_SAVE_DIR/gitksan-challenge_test.src

ONE_SOURCE_SAVEPREF=/scratch/fsamir8/finetune_randinit/1_source

# init environment
cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate

# clear old result files out
[ -f $ONE_SOURCE_SAVEPREF/results_dev.txt] && rm $ONE_SOURCE_SAVEPREF/results_dev.txt
[ -f $ONE_SOURCE_SAVEPREF/results_standard_test.txt] && rm $ONE_SOURCE_SAVEPREF/results_standard_test.txt
[ -f $ONE_SOURCE_SAVEPREF/results_challenge_test.txt] && rm $ONE_SOURCE_SAVEPREF/results_challenge_test.txt

fairseq-interactive --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 10 --source-lang src \
--target-lang tgt $ONE_SOURCE_PREPROC_SAVE_DIR < $ONE_SOURCE_DEVSRC > $ONE_SOURCE_SAVEPREF/results_dev.txt

fairseq-interactive --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 10 --source-lang src \
--target-lang tgt $ONE_SOURCE_PREPROC_SAVE_DIR < $ONE_SOURCE_STANDARD_TEST > $ONE_SOURCE_SAVEPREF/results_standard_test.txt

fairseq-interactive --path $ONE_SOURCE_CHECKPT_PREFIX/checkpoint_best.pt --beam 5 --nbest 10 --source-lang src \
--target-lang tgt $ONE_SOURCE_PREPROC_SAVE_DIR < $ONE_SOURCE_CHALLENGE_TEST > $ONE_SOURCE_SAVEPREF/results_challenge_test.txt