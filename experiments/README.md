# Experiments & Evaluation
Here are some instructions to run & evaluate the two methods presented in our paper called `datewise` and `clust`.

Required paths:
```bash
DATASETS=<folder with all datasets>
RESULTS=<folder to store results>
```

Running & evaluating the `datewise` method on the `t17` dataset:
```bash
python experiments/evaluate.py \
	--dataset $DATASETS/t17 \
	--method datewise \
	--resources resources/datewise \
	--output $RESULTS/t17.datewise.json
```
This method has a supervised component - regression for ranking dates. The regression models were trained separately and are only loaded and used in this process. Note that for each topic in a dataset, a different model was trained and is selected in evaluation because are doing leave-one-out cross-validation.

Running & evaluating the `clust` method on the `t17` dataset:
```bash
python experiments/evaluate.py \
	--dataset $DATASETS/t17 \
	--method clust \
	--output $RESULTS/t17.clust.json
```

For the other datasets, simply replace `t17` with `crisis` or `entities`.
