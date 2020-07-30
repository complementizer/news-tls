import os
import argparse
import arrow
import pathlib
import subprocess
import collections
import shutil
from news_tls import utils


def write_input_articles(articles, out_dir):
    utils.force_mkdir(out_dir)
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).datetime.date()
        date_to_articles[date].append(a)

    for date in sorted(date_to_articles):
        utils.force_mkdir(out_dir / str(date))
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = out_dir / str(date) / '{}.txt'.format(a['id'])
            with open(fpath, 'w') as f:
                f.write(a['text'])


def delete_input_articles(articles, out_dir):
    date_to_articles = collections.defaultdict(list)
    for a in articles:
        date = arrow.get(a['time']).datetime.date()
        date_to_articles[date].append(a)

    for date in sorted(date_to_articles):
        date_articles = date_to_articles[date]
        for a in date_articles:
            fpath = out_dir / str(date) / '{}.txt'.format(a['id'])
            if os.path.exists(fpath):
                os.remove(fpath)


def heideltime_preprocess(dataset_dir, heideltime_path):
    apply_heideltime = heideltime_path / 'apply-heideltime.jar'
    heideltime_config = heideltime_path / 'config.props'

    for topic in os.listdir(dataset_dir):
        print('TOPIC:', topic)

        articles = utils.read_jsonl_gz(dataset_dir / topic / 'articles.tokenized.jsonl.gz')

        out_dir = dataset_dir / topic / 'time_annotated'
        utils.force_mkdir(out_dir)
        write_input_articles(articles, out_dir)

        subprocess.run([
            'java',
            '-jar',
            str(apply_heideltime),
            str(heideltime_config),
            str(out_dir),
            'txt'
        ])

        delete_input_articles(articles, out_dir)


def main(args):
    dataset_dir = pathlib.Path(args.dataset)
    heideltime_path = pathlib.Path(args.heideltime)
    if not dataset_dir.exists():
        raise FileNotFoundError('dataset not found')
    if not heideltime_path.exists():
        raise FileNotFoundError('heideltime not found')

    heideltime_preprocess(dataset_dir, heideltime_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heideltime', required=True,
                        help='location of heideltime software')
    parser.add_argument('--dataset', required=True, help='dataset directory')
    main(parser.parse_args())
