import os
import argparse
import pathlib
import spacy
from news_tls import utils


def tokenize_dataset(root, spacy_model):
    nlp = spacy.load(spacy_model)

    for topic in sorted(os.listdir(root)):
        print('TOPIC:', topic)
        if os.path.exists(root / topic / 'articles.jsonl.gz'):
            articles = list(utils.read_jsonl_gz(root / topic / 'articles.jsonl.gz'))
        elif os.path.exists(root / topic / 'articles.jsonl'):
            articles = list(utils.read_jsonl(root / topic / 'articles.jsonl'))
        else:
            continue

        jsonl_out_path = root / topic / 'articles.tokenized.jsonl'
        out_batch = []
        for i, a in enumerate(articles):

            tokenized_doc = ''
            doc = nlp(a['text'])
            for sent in doc.sents:
                tokens = [tok.text for tok in sent if not tok.text.isspace()]
                tokenized_doc += ' '.join(tokens) + '\n'
            a['text'] = tokenized_doc.strip()
            out_batch.append(a)

            if i % 100 == 0:
                utils.write_jsonl(out_batch, jsonl_out_path, override=False)
                out_batch = []
                print(i)

        utils.write_jsonl(out_batch, jsonl_out_path, override=False)

        gz_out_path = root / topic / 'articles.tokenized.jsonl.gz'
        utils.gzip_file(jsonl_out_path, gz_out_path, delete_old=True)



def main(args):
    dataset_dir = pathlib.Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    tokenize_dataset(dataset_dir, args.spacy_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    main(parser.parse_args())
