#!/bin/bash

#TODO: add a parameter for where to move the generated hallucinated file
cd $WORKING_DIR/inflection
python augment.py sample-data/ gitksan 
mv sample-data/gitksan-hall "$1"