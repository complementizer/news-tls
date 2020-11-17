#!bash
DATASET=../../data/datasets/test
RESULTS=../../results
python experiments/run_without_eval.py --dataset $DATASET --method datewise --model resources/datewise/supervised_date_ranker.entities.pkl --output $RESULTS/test/datewise.json
