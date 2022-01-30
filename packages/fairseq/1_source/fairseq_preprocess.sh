#!/bin/bash
#SBATCH --time 00:05:00
#SBATCH --mem=1G
#SBATCH --job-name=TransformerBinarizeData
#SBATCH --output=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fsamir@mail.ubc.ca

WORKING_DIR="/project/rrg-msilfver/fsamir8"
cd $WORKING_DIR
source py3env/bin/activate
PREFIX=$WORKING_DIR/gitksan-data/data/spreadsheets/standard_challenge_split/type_split/cross_product_source/fairseq/gitksan
mkdir -p /project/rrg-msilfver/fsamir8/gitksan-data/results/1_source
1_SOURCE_OUT=$WORKING_DIR/gitksan-data/results/1_source
echo $1_SOURCE_OUT
echo $PREFIX
fairseq-preprocess --trainpref $PREFIX-train --validpref $PREFIX-dev --testpref $PREFIX-standard_test,$PREFIX-challenge_test --source-lang src \
   --target-lang tgt --destdir $1_SOURCE_OUT --cpu