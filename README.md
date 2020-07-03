# News Timeline Summarization
Repo the ACL 2020 paper [Examining the State-of-the-Art in News Timeline Summarization](https://arxiv.org/abs/2005.10107)

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
Checkout `news_tls/explore_dataset.py` to see how to load the provided datasets.
