import random
import datetime
import collections
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from news_tls import data, utils, summarizers


random.seed(42)


class DatewiseTimelineGenerator():
    def __init__(self,
                 date_ranker=None,
                 summarizer=None,
                 sent_collector=None,
                 clip_sents=5,
                 pub_end=2,
                 key_to_model=None):

        self.date_ranker = date_ranker or MentionCountDateRanker()
        self.sent_collector = sent_collector or PM_Mean_SentenceCollector(
            clip_sents, pub_end)
        self.summarizer = summarizer or summarizers.CentroidOpt()
        self.key_to_model = key_to_model

    def predict(self,
                collection,
                max_dates=10,
                max_summary_sents=1,
                ref_tl=None,
                input_titles=False,
                output_titles=False,
                output_body_sents=True):
        print('vectorizer...')
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        vectorizer.fit([s.raw for a in collection.articles() for s in a.sentences])

        print('date ranking...')
        ranked_dates = self.date_ranker.rank_dates(collection)

        start = collection.start.date()
        end = collection.end.date()
        ranked_dates = [d for d in ranked_dates if start <= d <= end]

        print('candidates & summarization...')
        dates_with_sents = self.sent_collector.collect_sents(
            ranked_dates,
            collection,
            vectorizer,
            include_titles=input_titles,
        )

        def sent_filter(sent):
            """
            Returns True if sentence is allowed to be in a summary.
            """
            lower = sent.raw.lower()
            if not any([kw in lower for kw in collection.keywords]):
                return False
            elif not output_titles and sent.is_title:
                return False
            elif not output_body_sents and not sent.is_sent:
                return False
            else:
                return True

        timeline = []
        l = 0
        for i, (d, d_sents) in enumerate(dates_with_sents):
            if l >= max_dates:
                break

            summary = self.summarizer.summarize(
                d_sents,
                k=max_summary_sents,
                vectorizer=vectorizer,
                filter=sent_filter
            )
            if summary:
                time = datetime.datetime(d.year, d.month, d.day)
                timeline.append((time, summary))
                l += 1

        timeline.sort(key=lambda x: x[0])
        return data.Timeline(timeline)

    def load(self, ignored_topics):
        key = ' '.join(sorted(ignored_topics))
        if self.key_to_model:
            self.date_ranker.model = self.key_to_model[key]


################################ DATE RANKING ##################################

class DateRanker:
    def rank_dates(self, collection, date_buckets):
        raise NotImplementedError


class RandomDateRanker(DateRanker):
    def rank_dates(self, collection):
        dates = [a.time.date() for a in collection.articles()]
        random.shuffle(dates)
        return dates


class MentionCountDateRanker(DateRanker):
    def rank_dates(self, collection):
        date_to_count = collections.defaultdict(int)
        for a in collection.articles():
            for s in a.sentences:
                d = s.get_date()
                if d:
                    date_to_count[d] += 1
        ranked = sorted(date_to_count.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]


class PubCountDateRanker(DateRanker):
    def rank_dates(self, collection):
        dates = [a.time.date() for a in collection.articles()]
        counts = collections.Counter(dates)
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]


class SupervisedDateRanker(DateRanker):
    def __init__(self, model=None, method='classification'):
        self.model = model
        self.method = method
        if method not in ['classification', 'regression']:
            raise ValueError('method must be classification or regression')

    def rank_dates(self, collection):
        dates, X = self.extract_features(collection)
        X = normalize(X, norm='l2', axis=0)
        if self.method == 'classification':
            Y = [y[1] for y in self.model['model'].predict_proba(X)]
        else:
            Y = self.model['model'].predict(X)
        scored = sorted(zip(dates, Y), key=lambda x: x[1], reverse=True)
        ranked = [x[0] for x in scored]
        # for d, score in scored[:16]:
        #     print(d, score)
        return ranked

    def extract_features(self, collection):
        date_to_stats = self.extract_date_statistics(collection)
        dates = sorted(date_to_stats)
        X = []
        for d in dates:
            feats = [
                date_to_stats[d]['sents_total'],
                date_to_stats[d]['sents_before'],
                date_to_stats[d]['sents_after'],
                date_to_stats[d]['docs_total'],
                date_to_stats[d]['docs_before'],
                date_to_stats[d]['docs_after'],
                date_to_stats[d]['docs_published'],
            ]
            X.append(np.array(feats))
        X = np.array(X)
        return dates, X

    def extract_date_statistics(self, collection):
        default = lambda: {
            'sents_total': 0,
            'sents_same_day': 0,
            'sents_before': 0,
            'sents_after': 0,
            'docs_total': 0,
            'docs_same_day': 0,
            'docs_before': 0,
            'docs_after': 0,
            'docs_published': 0
        }
        date_to_feats = collections.defaultdict(default)
        for a in collection.articles():
            pub_date = a.time.date()
            mentioned_dates = []
            for s in a.sentences:
                if s.time and s.time_level == 'd':
                    d = s.time.date()
                    date_to_feats[d]['sents_total'] += 1
                    if d < pub_date:
                        date_to_feats[d]['sents_before'] += 1
                    elif d > pub_date:
                        date_to_feats[d]['sents_after'] += 1
                    else:
                        date_to_feats[d]['sents_same_day'] += 1
                    mentioned_dates.append(d)
            for d in sorted(set(mentioned_dates)):
                date_to_feats[d]['docs_total'] += 1
                if d < pub_date:
                    date_to_feats[d]['docs_before'] += 1
                elif d > pub_date:
                    date_to_feats[d]['docs_after'] += 1
                else:
                    date_to_feats[d]['docs_same_day'] += 1
        return date_to_feats


############################## CANDIDATE SELECTION #############################


class M_SentenceCollector:
    def collect_sents(self, ranked_dates, collection, vectorizer, include_titles):
        date_to_ment = collections.defaultdict(list)
        for a in collection.articles():
            for s in a.sentences:
                ment_date = s.get_date()
                if ment_date:
                    date_to_ment[ment_date].append(s)
        for d in ranked_dates:
            if d in date_to_ment:
                d_sents = date_to_ment[d]
                if d_sents:
                    yield (d, d_sents)


class P_SentenceCollector:
    def __init__(self, clip_sents=5, pub_end=2):
        self.clip_sents = clip_sents
        self.pub_end = pub_end

    def collect_sents(self, ranked_dates, collection, vectorizer, include_titles):
        date_to_pub = collections.defaultdict(list)
        for a in collection.articles():
            pub_date = a.time.date()
            if include_titles:
                for k in range(self.pub_end):
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    if a.title_sentence:
                        date_to_pub[pub_date2].append(a.title_sentence)
            for s in a.sentences[:self.clip_sents]:
                for k in range(self.pub_end):
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    date_to_pub[pub_date2].append(s)
        for d in ranked_dates:
            if d in date_to_pub:
                d_sents = date_to_pub[d]
                if d_sents:
                    yield (d, d_sents)


class PM_All_SentenceCollector:
    def __init__(self, clip_sents=5, pub_end=2):
        self.clip_sents = clip_sents
        self.pub_end = pub_end

    def collect_sents(self, ranked_dates, collection, vectorizer, include_titles):
        date_to_sents = collections.defaultdict(list)
        for a in collection.articles():
            pub_date = a.time.date()
            if include_titles:
                for k in range(self.pub_end):
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    if a.title_sentence:
                        date_to_sents[pub_date2].append(a.title_sentence)
            for j, s in enumerate(a.sentences):
                ment_date = s.get_date()
                if ment_date:
                    date_to_sents[ment_date].append(s)
                elif j <= self.clip_sents:
                    for k in range(self.pub_end):
                        pub_date2 = pub_date - datetime.timedelta(days=k)
                        date_to_sents[pub_date2].append(s)
        for d in ranked_dates:
            if d in date_to_sents:
                d_sents = date_to_sents[d]
                if d_sents:
                    yield (d, d_sents)


class PM_Mean_SentenceCollector:
    def __init__(self, clip_sents=5, pub_end=2):
        self.clip_sents = clip_sents
        self.pub_end = pub_end

    def collect_sents(self, ranked_dates, collection, vectorizer, include_titles):
        date_to_pub, date_to_ment = self._first_pass(
            collection, include_titles)
        for d, sents in self._second_pass(
            ranked_dates, date_to_pub, date_to_ment, vectorizer):
            yield d, sents

    def _first_pass(self, collection, include_titles):
        date_to_ment = collections.defaultdict(list)
        date_to_pub = collections.defaultdict(list)
        for a in collection.articles():
            pub_date = a.time.date()
            if include_titles:
                for k in range(self.pub_end):
                    pub_date2 = pub_date - datetime.timedelta(days=k)
                    if a.title_sentence:
                        date_to_pub[pub_date2].append(a.title_sentence)
            for j, s in enumerate(a.sentences):
                ment_date = s.get_date()
                if ment_date:
                    date_to_ment[ment_date].append(s)
                elif j <= self.clip_sents:
                    for k in range(self.pub_end):
                        pub_date2 = pub_date - datetime.timedelta(days=k)
                        date_to_pub[pub_date2].append(s)
        return date_to_pub, date_to_ment

    def _second_pass(self, ranked_dates, date_to_pub, date_to_ment, vectorizer):

        for d in ranked_dates:
            ment_sents = date_to_ment[d]
            pub_sents = date_to_pub[d]
            selected_sents = []

            if len(ment_sents) > 0 and len(pub_sents) > 0:
                X_ment = vectorizer.transform([s.raw for s in ment_sents])
                X_pub = vectorizer.transform([s.raw for s in pub_sents])

                C_ment = sparse.csr_matrix(X_ment.sum(0))
                C_pub = sparse.csr_matrix(X_pub.sum(0))
                ment_weight = 1 / len(ment_sents)
                pub_weight = 1 / len(pub_sents)
                C_mean = (ment_weight * C_ment + pub_weight * C_pub)
                _, indices = C_mean.nonzero()

                C_date = sparse.lil_matrix(C_ment.shape)
                for i in indices:
                    v_pub = C_pub[0, i]
                    v_ment = C_ment[0, i]
                    if v_pub == 0 or v_ment == 0:
                        C_date[0, i] = 0
                    else:
                        C_date[0, i] = pub_weight * v_pub + ment_weight * v_ment

                ment_sims = cosine_similarity(C_date, X_ment)[0]
                pub_sims = cosine_similarity(C_date, X_pub)[0]
                all_sims = np.concatenate([ment_sims, pub_sims])

                cut = detect_knee_point(sorted(all_sims, reverse=True))
                thresh = all_sims[cut]

                for s, sim in zip(ment_sents, ment_sims):
                    if sim > 0 and sim > thresh:
                        selected_sents.append(s)
                for s, sim in zip(pub_sents, pub_sims):
                    if sim > 0 and sim > thresh:
                        selected_sents.append(s)

                if len(selected_sents) == 0:
                    selected_sents = ment_sents + pub_sents
            elif len(ment_sents) > 0:
                selected_sents = ment_sents
            elif len(pub_sents) > 0:
                selected_sents = pub_sents
            yield d, selected_sents


def detect_knee_point(values):
    """
    From:
    https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    # get coordinates of all the points
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    # get the first point
    first_point = all_coords[0]
    # get vector between first and last point - this is the line
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))
    vec_from_first = all_coords - first_point
    scalar_prod = np.sum(
        vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # distance to line is the norm of vec_to_line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # knee/elbow is the point with max distance value
    best_idx = np.argmax(dist_to_line)
    return best_idx
