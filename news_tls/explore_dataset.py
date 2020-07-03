import os
import argparse
import collections
from news_tls import utils
from news_tls.data import (Dataset,
                           truncate_timelines,
                           get_input_time_span,
                           get_average_summary_length)
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from pprint import pprint


def explore_dataset(dataset, trunc_timelines, time_span_extension):

    for collection in dataset.collections:
        print('topic:', collection.name)

        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]

        if trunc_timelines:
            ref_timelines = truncate_timelines(ref_timelines, collection)

        # each collection/topic can have multiple reference timelines
        for i, ref_timeline in enumerate(ref_timelines):
            ref_dates = sorted(ref_timeline.dates_to_summaries)

            # depending on the reference timeline, we set the time range in
            # article collection differently
            start, end = get_input_time_span(ref_dates, time_span_extension)
            collection.start = start
            collection.end = end

            #utils.plot_date_stats(collection, ref_dates)

            l = len(ref_dates)
            k = get_average_summary_length(ref_timeline)

            print(f'timeline:{i}, k:{k}, l:{l}')
        print()


def main(args):
    dataset_name = os.path.basename(args.dataset)
    dataset = Dataset(args.dataset)

    # these are settings we only apply to our new dataset (entities) but not to
    # crisis/t17 to keep these comparable to previous work
    if dataset_name == 'entities':
        explore_dataset(dataset, trunc_timelines=True, time_span_extension=7)
    else:
        explore_dataset(dataset, trunc_timelines=False, time_span_extension=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    main(parser.parse_args())
