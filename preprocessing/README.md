# to process the data from the nyt raw data

```bash
python preprocess_nyt.py --dataset_dir ../../../data/datasets/test --keywords apple,jobs
```
We provide the function to find a collection of the document from the specific kewords list and we provide an API for search the document information by parsing the id information.

> id is a code consisted of the Year,Month,Date,Slug, such as 20030404-LTR232202.

Please notice that Slug column is almost an unique identifier for one document in one day except for the death notice. Deathnotice' slug is like DN030403 which indicate that Death Notice in 2003-03-04. Since the only case using the death notice is a biography of one person, and which is easy to select manually.

we use simple query method to test whether the article abstract has the keyword and take it into jsonlines, where one json format is in the following.

```json
{'id': '20030404-04PHON$03',
 'time': '20030404T000000',
 'text': 'Stephen Holden reviews movie Phone Booth, directed by Joel Schumacher; Colin Farrell and Kiefer Sutherland star; photos (M)'}
```

for following preprocessing, there is a bash provided.

```bash
bash preprocess.sh
```

# Preprocessing
Here are some instructions for preprocessing a new dataset in the same way as the datasets used by us. [Tilse](https://github.com/smartschat/tilse) needs to be downloaded. It contains the HeidelTime tool in [this folder](https://github.com/smartschat/tilse/tree/master/tilse/tools/heideltime).

To start, we assume you have a directory for the new dataset with the following file structure:
```
dataset/
└── topic
    ├── articles.jsonl
    ├── timelines.jsonl
    └── keywords.json
```
Each line in `articles.jsonl` is a document or article in JSON format, with the following fields: `id`, `time`, `text` and optionally `title`. The document IDs must be unique. `timelines.jsonl` is only needed if you actually have ground-truth timelines. `time` is the publication time of an article. It should be at the day level or more specific and can be written in any format as long as it is recognised by `arrow.get` in the [arrow library](https://github.com/arrow-py/arrow).

You can download a "raw" mock dataset from [here](https://drive.google.com/drive/folders/1PTmON99RDUGqfkMqEDfQhmW_1N4926HW?usp=sharing).

We need to define two paths:
```bash
DATASET=<your dataset folder>
HEIDELTIME=<heideltime folder from above>
```

We then run these preprocessing steps (following Tilse):

```bash
python preprocess_tokenize.py --dataset $DATASET
python preprocess_heideltime.py --dataset $DATASET --heideltime $HEIDELTIME
python preprocess_spacy.py --dataset $DATASET
```
Note that the second step (running HeidelTime) is the slowest and is also responsible some amount of articles being removed in the last step if they cannot be parsed from HeidelTime's output.

Loading the preprocessed dataset:
```python
from news_tls.data import Dataset

dataset = Dataset('<path to dataset>')
for col in dataset.collections:
    print(col.name) # topic name
    print(col.keywords) # topic keywords
    for a in col.articles(): # articles collection of this topic
        pass # do something with the articles
```
