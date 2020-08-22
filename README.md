# News Timeline Summarization
Data & code for the ACL 2020 paper Examining the State-of-the-Art in News Timeline Summarization ([paper](https://www.aclweb.org/anthology/2020.acl-main.122.pdf),  [slides](acl20-slides.pdf)).

### Updates
Available
* all datasets
* methods & evaluation code
* preprocessing instructions for new datasets

Planned
* instructions to train date ranking models
* more user-friendly fast TLS version to run on unpreprocessed data

### Datasets

All datasets used in our experiments are [available here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing), including:
* T17
* Crisis
* Entities
### Library installation
The `news-tls` library contains tools for loading TLS datasets and running TLS methods.
To install, run:
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.

### Loading a dataset
Check out [news_tls/explore_dataset.py](news_tls/explore_dataset.py) to see how to load the provided datasets.

### Running methods & evaluation
Check out [experiments here](experiments).

### Format & preprocess your own dataset
If you have a new dataset yourself and want to use preprocess it as the datasets above, check out the [preprocessing steps here](preprocessing).

### Citation
```
@inproceedings{gholipour-ghalandari-ifrim-2020-examining,
    title = "Examining the State-of-the-Art in News Timeline Summarization",
    author = "Gholipour Ghalandari, Demian  and
      Ifrim, Georgiana",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.122",
    pages = "1322--1334",
}
```
