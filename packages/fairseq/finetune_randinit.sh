#!/bin/bash
​
#SBATCH --account=def-msilfver
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --mail-user=icoates1@mail.ubc.ca
#SBATCH --job-name=lstm_randinit_train
​
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NVIDIA/cuda-9.0/lib64
​
# define paths
PREPROCESS=/project/def-msilfver/Fairseq-sockeye/context/context_1-preprocess
SAVEPREF=/scratch/ecoates/finetune_randinit
​
# init environment
cd /project/def-msilfver/ecoates
source env/bin/activate
​
export CUDA_VISIBLE_DEVICES=0
​
for i in {1..10}; do
	fairseq-train $PREPROCESS \
		--source-lang src \
		--target-lang trg \
		--save-dir $SAVEPREF/$i_checkpts \
		--seed $i \
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
		--clip-norm 0 \
		--lr-scheduler inverse_sqrt \
		--warmup-updates 400 \
		--warmup-init-lr 1e-7 --lr 0.0005 \
		--max-tokens 4000 \
		--update-freq 1 \
		--max-epoch 500 \
		--ddp-backend=no_c10d \
		--save-interval 4 \
		--save-interval-updates 1000 --keep-interval-updates 20 \
		--log-format json --log-interval 20
done