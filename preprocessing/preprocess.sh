#!bash
DATASET=../../../data/datasets/test
HEIDELTIME=../../../tilse-master/tilse/tools/heideltime
python preprocess_tokenize.py --dataset $DATASET
python preprocess_heideltime.py --dataset $DATASET --heideltime $HEIDELTIME
python preprocess_spacy.py --dataset $DATASET