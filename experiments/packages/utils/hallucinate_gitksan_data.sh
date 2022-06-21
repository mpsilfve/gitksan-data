#!/bin/bash

#TODO: add a parameter for where to move the generated hallucinated file
cd /project/rrg-msilfver/fsamir8/inflection
python augment.py sample-data/ gitksan 
mv /project/rrg-msilfver/fsamir8/inflection/sample-data/gitksan-hall /project/rrg-msilfver/fsamir8/gitksan-data/"$1"
sed -i -E "s/IN;/IN:/g" /project/rrg-msilfver/fsamir8/gitksan-data/"$1"/gitksan-hall 
sed -i -E "s/OUT;/OUT:/g" /project/rrg-msilfver/fsamir8/gitksan-data/"$1"/gitksan-hall 
sed -i -E "s/;II/\.II/g" /project/rrg-msilfver/fsamir8/gitksan-data/"$1"/gitksan-hall 