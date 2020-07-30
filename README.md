# News Timeline Summarization
Data & code for the ACL 2020 paper *Examining the State-of-the-Art in News Timeline Summarization ([paper](https://www.aclweb.org/anthology/2020.acl-main.122.pdf),  [slides](acl20-slides.pdf)).

### Updates
Datasets are available, code for methods will follow soon.

### Datasets

All datasets used in our experiments are [available here](https://drive.google.com/drive/folders/1gDAF5QZyCWnF_hYKbxIzOyjT6MSkbQXu?usp=sharing), including:
* T17
* Crisis
* Entities
### Installation
Install requirements & the `news_tls` library.
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.

### Loading a dataset
Checkout [news_tls/explore_dataset.py](news_tls/explore_dataset.py) to see how to load the provided datasets.

### Format & preprocess your own dataset
If you have a new dataset yourself and want to use preprocess it as the datasets above, checkout the [preprocessing steps here](preprocessing).

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
