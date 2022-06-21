#!/bin/bash

# echo "Cross-table standard scores"
# python main_transformer_experiments.py --eval_fairseq_cross_table_random
# echo "Cross-table MBR scores"
# python main_transformer_experiments.py --eval_fairseq_cross_table_random_mbr
# echo "Cross-table platt-scaled MBR scores"
# python main_transformer_experiments.py --eval_fairseq_cross_table_random_mbr_platt_scaled

echo "hallucination standard scores"
python main_transformer_experiments.py --eval_fairseq_hallucination_random
echo "hallucination MBR scores"
python main_transformer_experiments.py --eval_fairseq_hallucination_random_mbr
echo "hallucination platt-scaled MBR scores"
python main_transformer_experiments.py --eval_fairseq_hallucination_random_mbr_platt_scaled