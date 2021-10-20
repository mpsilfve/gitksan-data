#!/bin/bash
​
N=3
​
PREF=../data/datasets/context_$N/context_$N
OUT=../data/models/context
​
fairseq-preprocess --trainpref $PREF-train --validpref $PREF-dev --testpref $PREF-test --source-lang src \
  --joined-dictionary --target-lang trg --destdir $OUT/context_$N-preprocess --cpu