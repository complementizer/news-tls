import argparse
from pathlib import Path
from news_tls import utils, data, datewise, clust, summarizers
from pprint import pprint



def run(tls_model, dataset, outpath):

    n_topics = len(dataset.collections)
    outputs = []

    for i, collection in enumerate(dataset.collections):
        topic = collection.name
        times = [a.time for a in collection.articles()]
        # setting start, end, L, K manually instead of from ground-truth
        collection.start = min(times)
        collection.end = max(times)
        l = 8 # timeline length (dates)
        k = 1 # number of sentences in each summary

        timeline = tls_model.predict(
            collection,
            max_dates=l,
            max_summary_sents=k,

        )

        print('*** TIMELINE ***')
        utils.print_tl(timeline)

        outputs.append(timeline.to_dict())

    if outpath:
        utils.write_json(outputs, outpath)


def main(args):

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.method == 'datewise':
        # load regression models for date ranking
        key_to_model = utils.load_pkl(args.model)
        models = list(key_to_model.values())
        date_ranker = datewise.SupervisedDateRanker(method='regression')
        # there are multiple models (for cross-validation),
        # we just an arbitrary model, the first one
        date_ranker.model = models[0]
        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=2, pub_end=2)
        summarizer = summarizers.CentroidOpt()
        system = datewise.DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model = key_to_model
        )

    elif args.method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=2,
            unique_dates=True,
        )
    else:
        raise ValueError(f'Method not found: {args.method}')


    run(system, dataset, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--model', default=None,
        help='model for date ranker')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())
