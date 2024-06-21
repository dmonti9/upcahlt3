#! /bin/bash

AHLT=..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 8 final+glove+weights+LReLU

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py final+glove+weights+LReLU $AHLT/data/devel > final+glove+weights+LReLU.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel final+glove+weights+LReLU.out > final+glove+weights+LReLU.stats
