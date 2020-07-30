import os
import pathlib
import argparse
import arrow
import spacy
import datetime
import collections
import codecs
from xml.etree import ElementTree
from news_tls import utils
from news_tls.data import Token, Sentence, Article
from pprint import pprint


def extract_time_tag_value(time_tag):
    value = [(None, None)]

    if 'type' not in time_tag.attrib:
        return value
    elif time_tag.attrib['type'] == 'DATE':
        formats = ['%Y-%m-%d', '%Y-%m', '%Y']
    elif time_tag.attrib['type'] == 'TIME':
        formats = ['%Y-%m-%dT%H:%M', '%Y-%m-%dTMO', '%Y-%m-%dTEV',
                   '%Y-%m-%dTNI', '%Y-%m-%dTAF']
    else:
        return value

    for format in formats:
        try:
            time = datetime.datetime.strptime(
                time_tag.attrib['value'], format)
            value = [(time, format)]
        except:
            pass
    return value


def parse_timeml_doc(raw):

    # cleanup heideltime bugs
    replace_pairs = [
        ("T24", "T12"),
        (")TMO", "TMO"),
        (")TAF", "TAF"),
        (")TEV", "TEV"),
        (")TNI", "TNI"),
    ]
    for old, new in replace_pairs:
        raw = raw.replace(old, new)

    tokens = []
    time_values = []

    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as e:
        return None, None

    tokens.extend(root.text.split())
    time_values.extend([(None, None)] * len(tokens))

    for time_tag in root:
        if time_tag.text is None:
            continue
        split_text = time_tag.text.split()
        tokens.extend(split_text)
        value = extract_time_tag_value(time_tag)
        time_values.extend(value * len(split_text))
        split_tail = time_tag.tail.split()
        tokens.extend(split_tail)
        time_values.extend([(None, None)] * len(split_tail))

    return tokens, time_values


def read_articles(articles, tmp_dir):
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).date()
        date_to_articles[date].append(a)
    for date in sorted(date_to_articles):
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = tmp_dir / str(date) / '{}.txt.timeml'.format(a['id'])
            if os.path.exists(fpath):
                with codecs.open(fpath, 'r', encoding='utf-8') as f:
                    raw = f.read()
                yield a, raw


def preprocess_title(title, pub_time, nlp):
    doc = nlp(title)
    token_objects = []
    for token in doc:
        token_object = Token(
            token.orth_,
            token.lemma_,
            token.tag_,
            token.ent_type_,
            token.ent_iob_,
            token.dep_,
            token.head.i,
            None,
            None,
        )
        token_objects.append(token_object)
    title_object = Sentence(title, token_objects, pub_time, None, None)
    return title_object


def preprocess_article(old_article, timeml_raw, nlp):
    tokens, time_values = parse_timeml_doc(timeml_raw)

    if tokens is None:
        return None

    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    nlp.tagger(doc)
    nlp.entity(doc)
    nlp.parser(doc)

    token_objects = []
    for token in doc:
        token_object = Token(
            token.orth_,
            token.lemma_,
            token.tag_,
            token.ent_type_,
            token.ent_iob_,
            token.dep_,
            token.head.i,
            time_values[token.i][0],
            time_values[token.i][1],
        )
        token_objects.append(token_object)

    sentence_objects = []
    for sent in doc.sents:
        sent_tokens = token_objects[sent.start:sent.end]
        times = [tok.time for tok in sent_tokens if tok.time]
        if times:
            time = times[0]
        else:
            time = None

        pub_time = arrow.get(old_article['time'])
        sent_object = Sentence(str(sent), sent_tokens, pub_time, time, None)
        sentence_objects.append(sent_object)

    raw_title = old_article.get('title')
    if raw_title:
        title_object = preprocess_title(raw_title, pub_time, nlp)
    else:
        title_object = None

    new_article = Article(
        title=raw_title,
        text=old_article['text'],
        time=old_article['time'],
        id=old_article.get('id'),
        sentences=sentence_objects,
        title_sentence=title_object
    )
    return new_article


def preprocess_dataset(root, nlp):

    for topic in sorted(os.listdir(root)):
        print('TOPIC:', topic)

        article_path = root / topic / 'articles.tokenized.jsonl.gz'
        articles = utils.read_jsonl_gz(article_path)
        h_output_dir = root / topic / 'time_annotated'
        out_path = root / topic / 'articles.preprocessed.jsonl'
        out_batch = []
        i = 0

        for old_a, timeml_raw in read_articles(articles, h_output_dir):
            a = preprocess_article(old_a, timeml_raw, nlp)

            if a:
                out_batch.append(a.to_dict())
            else:
                date = arrow.get(old_a['time']).date()
                print('cannot process:', date, old_a['id'])

            if i % 100 == 0:
                print('writing batch,', i, 'articles done')
                if i == 0:
                    utils.write_jsonl(out_batch, out_path, override=True)
                else:
                    utils.write_jsonl(out_batch, out_path, override=False)
                out_batch = []
            i += 1

        utils.write_jsonl(out_batch, out_path, override=False)
        gz_path = str(out_path) + '.gz'
        utils.gzip_file(inpath=out_path, outpath=gz_path, delete_old=True)


def main(args):
    dataset_dir = pathlib.Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    nlp = spacy.load(args.spacy_model)
    preprocess_dataset(dataset_dir, nlp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset directory')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    main(parser.parse_args())
