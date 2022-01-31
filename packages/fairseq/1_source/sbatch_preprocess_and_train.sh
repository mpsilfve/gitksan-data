#!/bin/bash

jid1=$(sbatch ./fairseq_preprocess.sh)
sbatch --dependency=afterany:$jid1 ./finetune_randinit.sh