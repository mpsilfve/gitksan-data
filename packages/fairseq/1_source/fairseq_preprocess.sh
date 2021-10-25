#!/bin/bash
​
PREF=../../data/spreadsheets/seen_unseen_split_w_root/fairseq/gitksan
OUT=../../results
​
fairseq-preprocess --trainpref $PREF-train --validpref $PREF-dev --testpref $PREF-seen-test,$PREF-unseen-test --source-lang src \
   --target-lang tgt --destdir $OUT --cpu
