#!/bin/bash
echo "Cleaning up tokenization for datapoints that were hallucinated"
cd "data/spreadsheets/seen_unseen_split_w_root_hall"
sed -i -E "s/IN\s/IN:/g" fairseq/gitksan-train.src
sed -i -E "s/OUT\s/OUT:/g" fairseq/gitksan-train.src
sed -i -E "s/\sII/\.II/g" fairseq/gitksan-train.src