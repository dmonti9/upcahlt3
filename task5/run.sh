#! /bin/bash

AHLT=..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 8 new

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py new $AHLT/data/devel > new.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel new.out > new.stats
