#!/bin/bash
​
#SBATCH --account=def-msilfver
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --mail-user=fsamir8@mail.ubc.ca
#SBATCH --job-name=lstm_randinit_train
#SBATCH --output=/scratch/fsamir8/finetune_randinit/gitksan.out
#SBATCH --error=/scratch/fsamir8/finetune_randinit/gitksan.error

​
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64
​
# define paths
PREPROCESS=/project/rrg-msilfver/fsamir8/gitksan-data/results
SAVEPREF=/scratch/fsamir8/finetune_randinit
​
# init environment
cd /project/rrg-msilfver/fsamir8
source py3env/bin/activate
​
export CUDA_VISIBLE_DEVICES=0
​
fairseq-train $PREPROCESS \
    --source-lang src \
    --target-lang tgt \
    --save-dir $SAVEPREF \
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
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --batch-size 400
    --clip-norm 0 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 400 \
    --warmup-init-lr 1e-7 --lr 0.0005 --min-lr 1e-9 \
    --no-epoch-checkpoints \
    --max-tokens 2000 \
    --update-freq 1 \
    --max-epoch 500 \
    --ddp-backend=no_c10d \
    --save-interval 4 \
    --save-interval-updates 1000 --keep-interval-updates 20 \
    --log-format json --log-interval 20 \