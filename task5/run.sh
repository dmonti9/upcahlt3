#! /bin/bash

AHLT=..

# train NN
echo "Training NN"
python3 train.py $AHLT/data/train $AHLT/data/devel 10 mymodel2-lwords-fc-feat-new

# run model on devel data and compute performance
echo "Predicting"
python3 predict.py mymodel2-lwords-fc-feat-new $AHLT/data/devel > devel-lwords-fc-feat-new.out 

# evaluate results
echo "Evaluating results..."
python3 $AHLT/util/evaluator.py NER $AHLT/data/devel devel-lwords-fc-feat-new.out > devel-lwords-fc-feat-new.stats
