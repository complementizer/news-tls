import os
import pathlib
import arrow
import datetime
import string
from spacy.lang.en.stop_words import STOP_WORDS
from collections import defaultdict
from tilse.data.timelines import Timeline as TilseTimeline
from news_tls import utils
from pprint import pprint

PUNCT_SET = set(string.punctuation)


def load_dataset(path):
    dataset = Dataset(path)
    return dataset


def load_article(article_dict):
    sentences = [load_sentence(x) for x in article_dict['sentences']]
    if article_dict.get('title_sentence'):
        title_sentence = load_sentence(article_dict['title_sentence'])
        title_sentence.is_title = True
    else:
        title_sentence = None
    fix_dependency_heads(sentences)
    time = arrow.get(article_dict['time']).datetime
    time = time.replace(tzinfo=None)
    return Article(
        article_dict['title'],
        article_dict['text'],
        time,
        article_dict['id'],
        sentences,
        title_sentence,
    )


def load_sentence(sent_dict):

    tokens = load_tokens(sent_dict['tokens'])
    pub_time = utils.strip_to_date(arrow.get(sent_dict['pub_time']))
    time = Sentence.get_time(tokens)
    time_level = None
    if time:
        time = arrow.get(time)
        time_format = Sentence.get_time_format(tokens)
        time_level = None
        if 'd' in time_format:
            time = datetime.datetime(time.year, time.month, time.day)
            time_level = 'd'
        elif ('m' in time_format) or ('y' in time_format):
            if 'm' in time_format:
                start, end = time.span('month')
                time_level = 'm'
            else:
                start, end = time.span('year')
                time_level = 'y'
            start = datetime.datetime(start.year, start.month, start.day)
            end = datetime.datetime(end.year, end.month, end.day)
            time = (start, end)

    return Sentence(
        sent_dict['raw'],
        tokens,
        pub_time,
        time,
        time_level
    )


def load_tokens(tokens_dict):
    token_dicts = decompress_dict_list(tokens_dict)
    tokens = []
    for token_ in token_dicts:
        token = Token(
            token_['raw'],
            token_['lemma'],
            token_['pos'],
            token_['ner_type'],
            token_['ner_iob'],
            token_['dep'],
            token_['head'],
            token_['time'],
            token_['time_format']
        )
        tokens.append(token)
    return tokens


def fix_dependency_heads(sentences):
    """
    Change from document to sentence-level head indices.
    """
    i = 0
    for s in sentences:
        for tok in s.tokens:
            tok.head -= i
        i += len(s.tokens)


class Token:
    def __init__(self, raw, lemma, pos, ner_type, ner_iob, dep, head, time,
                 time_format):
        self.raw = raw
        self.lemma = lemma
        self.pos = pos
        self.ner_type = ner_type
        self.ner_iob = ner_iob
        self.dep = dep
        self.head = head
        self.time = time
        self.time_format = time_format

    def to_dict(self):
        time = self.time.isoformat() if self.time else None
        return {
            'raw': self.raw,
            'lemma': self.lemma,
            'pos': self.pos,
            'ner_type': self.ner_type,
            'ner_iob': self.ner_iob,
            'dep': self.dep,
            'head': self.head,
            'time': time,
            'time_format': self.time_format
        }


class Sentence:
    def __init__(self, raw, tokens, pub_time, time, time_level, is_title=False):
        self.raw = raw
        self.tokens = tokens
        self.pub_time = pub_time
        self.time = time
        self.time_level = time_level
        self.is_title = is_title


    @staticmethod
    def get_time(tokens):
        for token in tokens:
            if token.time:
                return token.time
        return None

    @staticmethod
    def get_time_format(tokens):
        for token in tokens:
            if token.time_format:
                return token.time_format
        return None

    def get_date(self):
        if self.time_level == 'd':
            return self.time.date()
        else:
            return None

    def clean_tokens(self):
        tokens = [tok.raw.lower() for tok in self.tokens]
        tokens = [tok for tok in tokens if
                  (tok not in STOP_WORDS and tok not in PUNCT_SET)]
        return tokens

    def _group_entity(self, entity_tokens):
        surface_form = ' '.join([tok_.raw for tok_ in entity_tokens])
        type = entity_tokens[-1].ner_type
        return surface_form, type

    def get_entities(self, return_other=False):
        entities = []
        tmp_entity = []
        other = []
        for tok in self.tokens:
            if tok.ner_iob == 'B':
                if len(tmp_entity) > 0:
                    e = self._group_entity(tmp_entity)
                    entities.append(e)
                tmp_entity = [tok]
            elif tok.ner_iob == 'I':
                tmp_entity.append(tok)
            else:
                other.append(tok)
        if len(tmp_entity) > 0:
            e = self._group_entity(tmp_entity)
            entities.append(e)

        if return_other:
            return entities, other
        else:
            return entities

    def to_dict(self):
        if self.time:
            time = self.time.isoformat()
        else:
            time = None

        tokens = [tok.to_dict() for tok in self.tokens]
        tokens = compress_dict_list(tokens)

        return {
            'raw': self.raw,
            'tokens': tokens,
            'time': time,
            'pub_time': self.pub_time.isoformat(),
        }

class Article:
    '''
    Stores information about a news article.
    '''
    def __init__(self,
                 title,
                 text,
                 time,
                 id,
                 sentences=None,
                 title_sentence=None,
                 vector=None):
        self.title = title
        self.text = text
        self.time = time
        self.id = id
        self.sentences = sentences
        self.title_sentence = title_sentence
        self.vector = vector

    def to_dict(self):

        title_sent_dict = None
        if self.title_sentence:
            title_sent_dict = self.title_sentence.to_dict()

        return {
            'title': self.title,
            'text': self.text,
            'time': str(self.time),
            'id': self.id,
            'sentences': [s.to_dict() for s in self.sentences],
            'title_sentence': title_sent_dict,
            'vector': self.vector
        }


class Dataset:

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.topics = self._get_topics()
        self.collections = self._load_collections()

    def _get_topics(self):
        return sorted(os.listdir(self.path))

    def _load_collections(self):
        collections = []
        for topic in self.topics:
            topic_path = self.path / topic
            c = ArticleCollection(topic_path)
            collections.append(c)
        return collections


class ArticleCollection:
    def __init__(self, path, start=None, end=None):
        self.name = os.path.basename(path)
        self.path = pathlib.Path(path)
        self.keywords = utils.read_json(self.path / 'keywords.json')
        self.timelines = self._load_timelines()
        self.start = start
        self.end = end

    def _load_timelines(self):
        timelines = []
        path = self.path / 'timelines.jsonl'
        if not path.exists():
            return []
        for raw_tl in utils.read_jsonl(path):
            if raw_tl:
                tl_items = []
                for t, s in raw_tl:
                    t = self.normalise_time(arrow.get(t))
                    tl_items.append((t, s))
                tl = Timeline(tl_items)
                timelines.append(tl)
        return timelines

    def articles(self):
        path1 = self.path / 'articles.preprocessed.jsonl'
        path2 = self.path / 'articles.preprocessed.jsonl.gz'
        if path1.exists():
            articles = utils.read_jsonl(path1)
        else:
            articles = utils.read_jsonl_gz(path2)
        for a_ in articles:
            a = load_article(a_)
            t = self.normalise_time(a.time)
            if self.start and t < self.start:
                continue
            if self.end and t > self.end:
                break
            yield a

    def time_batches(self):
        articles = utils.read_jsonl_gz(self.path / 'articles.preprocessed.jsonl.gz')
        time = None
        batch = []
        for a_ in articles:
            a = load_article(a_)
            a_time = self.normalise_time(a.time)

            if self.start and a_time < self.start:
                continue

            if self.end and a_time > self.end:
                break

            if time and a_time > time:
                yield time, batch
                time = a_time
                batch = [a]
            else:
                batch.append(a)
                time = a_time
        yield time, batch

    def times(self):
        articles = utils.read_jsonl(self.path / 'articles.preprocessed.jsonl')
        times = []
        for a in articles:
            t = arrow.get(a['time']).datetime
            t = t.replace(tzinfo=None)
            times.append(t)
        return times

    def normalise_time(self, t):
        return datetime.datetime(t.year, t.month, t.day)


class Timeline:
    def __init__(self, items):
        self.items = sorted(items, key=lambda x: x[0])
        self.time_to_summaries = dict((t, s) for t, s in items)
        self.date_to_summaries = dict((t.date(), s) for t, s in items)
        self.times = sorted(self.time_to_summaries)

    def __getitem__(self, item):
        return self.time_to_summaries[item]

    def __len__(self):
        return len(self.items)

    def __str__(self):
        lines = []
        for t, summary in self.items:
            lines.append('[{}]'.format(t.date()))
            for sent in summary:
                lines.append(sent)
            lines.append('-'*50)
        return '\n'.join(lines)

    def to_dict(self):
        items = [(str(t), s) for (t, s) in self.items]
        return items


def compress_dict_list(dicts):
    keys = sorted(dicts[0].keys())
    data = []
    for d in dicts:
        values = [d[k] for k in keys]
        data.append(values)
    return {
        'keys': keys,
        'data': data
    }


def decompress_dict_list(x):
    dicts = []
    keys = x['keys']
    for values in x['data']:
        d = dict(zip(keys, values))
        dicts.append(d)
    return dicts


def truncate_timelines(ref_timelines_, collection):
    input_dates = [t.date() for t, _ in collection.time_batches()]
    input_date_set = set(input_dates)
    input_start = min(input_dates)
    input_end = max(input_dates)
    ref_timelines = []
    for tl in ref_timelines_:
        dates_to_summaries = tl.dates_to_summaries
        new_dates_to_summaries = {}
        for d, s in dates_to_summaries.items():
            if d >= input_start and d <= input_end:
                window_start = d + datetime.timedelta(days=-2)
                window_end = d + datetime.timedelta(days=+2)
                window = utils.get_date_range(window_start, window_end)
                if any([d2 in input_date_set for d2 in window]):
                    new_dates_to_summaries[d] = s
        tl = TilseTimeline(dates_to_summaries)
        ref_timelines.append(tl)
    return ref_timelines


def get_average_summary_length(ref_tl):
    lens = []
    for date, summary in ref_tl.dates_to_summaries.items():
        lens.append(len(summary))
    k = sum(lens) / len(lens)
    return round(k)


def get_input_time_span(ref_dates, extension):
    ref_start = utils.strip_to_date(min(ref_dates))
    ref_end = utils.strip_to_date(max(ref_dates))
    input_start = ref_start - datetime.timedelta(days=extension)
    input_end = ref_end + datetime.timedelta(days=extension)
    return input_start, input_end
