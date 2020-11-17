#!bash
DATASET=../../data/datasets/test
RESULTS=../../results
python experiments/run_without_eval.py --dataset $DATASET --method clust --output $RESULTS/test/clust.json
