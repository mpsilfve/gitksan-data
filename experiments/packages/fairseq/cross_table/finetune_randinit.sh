#!/bin/bash
#SBATCH --account=def-msilfver
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=03:00:00
#SBATCH --job-name=TransformerCrossTable
#SBATCH --output=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/cross_table/gitksan.error

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64
export CUDA_VISIBLE_DEVICES=0

WORKING_DIR="/project/rrg-msilfver/fsamir8"
CROSS_TABLE_PREPROC_SAVE_DIR=$WORKING_DIR/gitksan-data/results/cross_table
CROSS_TABLE_SAVEPREF=/scratch/fsamir8/finetune_randinit/cross_table

cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate

fairseq-train $CROSS_TABLE_PREPROC_SAVE_DIR \
    --no-epoch-checkpoints \
    --source-lang src \
    --target-lang tgt \
    --save-dir $CROSS_TABLE_SAVEPREF \
    --seed 1 \
    --arch transformer \
    --encoder-layers 4 \
    --decoder-layers 4 \
    --encoder-embed-dim 256 \
    --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 512 \
    --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --dropout 0.3 \
    --attention-dropout 0.1 \
    --relu-dropout 0 \
    --weight-decay 0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-betas '(0.9, 0.999)' \
    --batch-size 400 \
    --clip-norm 0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-7 --lr 0.001 --stop-min-lr 1e-9 \
    --keep-interval-updates 20 \
    --max-tokens 2000 \
    --max-update 20000 \
    --update-freq 1 \
    --max-epoch 1000 \
    --log-format json --log-interval 20 
EOF

