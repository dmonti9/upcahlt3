#! /bin/bash

AHLT=..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 10 mymodel4

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py mymodel4 $AHLT/data/devel > devel-4.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel devel-4.out > devel-4.stats
