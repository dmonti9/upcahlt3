#! /bin/bash

AHLT=..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 1 mymodel2-lwords-fc-ext

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py mymodel2-lwords-fc-ext $AHLT/data/devel > devel-lwords-fc-ext.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel devel-lwords-fc-ext.out > devel-lwords-fc-ext.stats
