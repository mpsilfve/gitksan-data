#!/bin/bash
#SBATCH --time 00:05:00
#SBATCH --mem=1G
#SBATCH --job-name=TransformerBinarizeData
#SBATCH --output=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.error
#SBATCH --mail-user=fsamir@mail.ubc.ca

WORKING_DIR="/project/rrg-msilfver/fsamir8"
cd $WORKING_DIR
source py3env/bin/activate
PREFIX=$WORKING_DIR/gitksan-data/data/spreadsheets/seen_unseen_split_w_root_cross_table/fairseq/gitksan
mkdir -p /project/rrg-msilfver/fsamir8/gitksan-data/results/cross_table
CROSS_TABLE_OUT=$WORKING_DIR/gitksan-data/results/cross_table
echo $CROSS_TABLE_OUT
echo $PREFIX
fairseq-preprocess --trainpref $PREFIX-train --validpref $PREFIX-dev --testpref $PREFIX-seen-test,$PREFIX-unseen-test --source-lang src \
   --target-lang tgt --destdir $CROSS_TABLE_OUT --cpu