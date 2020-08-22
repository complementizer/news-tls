# Experiments & Evaluation
### Reproduce evaluation
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
This method has a supervised component - regression for ranking dates. The regression models were trained separately and are only loaded and used in this process. Note that for each topic in a dataset, a different model was trained and is selected in evaluation because we are doing leave-one-out cross-validation.

Running & evaluating the `clust` method on the `t17` dataset:
```bash
python experiments/evaluate.py \
	--dataset $DATASETS/t17 \
	--method clust \
	--output $RESULTS/t17.clust.json
```

For the other datasets, simply replace `t17` with `crisis` or `entities`.

### Run methods without evaluation
This can be useful if you try the methods on some new dataset without any available ground-truth. Here is a [preprocessed mock dataset](https://drive.google.com/drive/folders/15xHJPOLc7v0yXSKjCNneELKYlwWYeNV0?usp=sharing) with one topic to try this out.

Running the clustering-based method:
```bash
DATASET=<pick some dataset>
python experiments/run_without_eval.py \
    --dataset $DATASET \
    --method clust \
```
When using the `datewise` method, a date ranking model is required. We can just pick one from the existing datasets:
```bash
DATASET=<pick some dataset>
python experiments/run_without_eval.py \
    --dataset $DATASET \
    --method datewise \
    --model resources/datewise/supervised_date_ranker.entities.pkl
```
You can change various settings of the methods and the timeline length and time span in [run_without_eval.py](run_without_eval.py).
