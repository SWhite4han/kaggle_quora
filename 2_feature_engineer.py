import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, canberra, minkowski, braycurtis
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import logging
import sys
import time
import os
import pickle

# Set parameters
stop_words = set(stopwords.words('english'))
common_start_words = ['why', 'what', 'how', "what's", 'do', 'does', 'is',
                      'can', 'which', 'if', 'i', 'are', 'where', 'who']


def set_logger(log_level=logging.INFO):
    """Configure the logger with log_level."""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=log_level,
        stream=sys.stderr)
    logger = logging.getLogger(__name__)
    # logging.getLogger('requests').setLevel(logging.WARNING)
    return logger


def word_len(s):
    return len(str(s))


def common_words(row):
    set1 = set(row['q1_split'])
    set2 = set(row['q2_split'])
    return len(set1.intersection(set2))


def common_words_unit(row):
    set1 = set(row['question1'])
    set2 = set(row['question2'])
    return len(set1.intersection(set2))


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        q1words[word] = 1
    for word in row['q2_split']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def word_match_share_stops(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        if word not in stops:
            q1words[word] = 1
    for word in row['q2_split']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        if word not in stops:
            q1words[word] = 1
    for word in row['q2_split']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        q1words[word] = 1
    for word in row['q2_split']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def calculate_tfidf(qs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(qs)
    idf = vectorizer.idf_
    dict_tfidf = dict(zip(vectorizer.get_feature_names(), idf))
    # log.info('\nMost common words and weights:')
    # log.info(sorted(dict_tfidf.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
    # log.info('\nLeast common words and weights: ')
    # log.info(sorted(dict_tfidf.items(), key=lambda x: x[1], reverse=True)[:10])
    # log.info(dict_tfidf.get('how', 0))
    return dict_tfidf


def jaccard(row):
    wic = set(row['q1_split']).intersection(set(row['q2_split']))
    uw = set(row['q1_split']).union(row['q2_split'])
    if len(uw) == 0:
        uw = [1]
    return len(wic) / len(uw)


def total_unique_words(row):
    return len(set(row['q1_split']).union(row['q2_split']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['q1_split']).union(row['q2_split']) if x not in stops])


def word_count(divided_s):
    return len(divided_s)


def wc_diff(row):
    return abs(len(row['q1_split']) - len(row['q2_split']))


def wc_ratio(row):
    l1 = len(row['q1_split']) * 1.0
    l2 = len(row['q2_split'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['q1_split'])) - len(set(row['q2_split'])))


def wc_ratio_unique(row):
    l1 = len(set(row['q1_split'])) * 1.0
    l2 = len(set(row['q2_split']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['q1_split']) if x not in stops]) - len(
        [x for x in set(row['q2_split']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['q1_split']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['q2_split']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['q1_split'] or not row['q2_split']:
        return np.nan
    return int(row['q1_split'][0] == row['q2_split'][0])


def same_end_word(row):
    if not row['q1_split'] or not row['q2_split']:
        return np.nan
    return int(row['q1_split'][-1] == row['q2_split'][-1])


def word_len_char(divided_s):
    return len(''.join(divided_s))


def len_char_diff(row):
    return abs(len(''.join(row['q1_split'])) - len(''.join(row['q2_split'])))


def char_ratio(row):
    l1 = len(''.join(row['q1_split']))
    l2 = len(''.join(row['q2_split']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['q1_split']) if x not in stops])) - len(
        ''.join([x for x in set(row['q2_split']) if x not in stops])))


def num_capital(s):
    return sum(1 for c in s if c.isupper())


def num_ques_mark(s):
    return sum(1 for c in s if c is '?')


def start_with(divided_s, start):
    if divided_s:
        return 1 if start == divided_s[0] else 0
    return 0


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def wmd(divided_s1, divided_s2):
    s1 = [w for w in divided_s1 if w not in stop_words]
    s2 = [w for w in divided_s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(divided_s1, divided_s2):
    s1 = [w for w in divided_s1 if w not in stop_words]
    s2 = [w for w in divided_s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    # words = str(s).lower().decode('utf-8')
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def clean_doc(s):
    # clean and tokenize document string
    raw = s.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if i not in stop_words]

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    return stemmed_tokens


def train_lda(dictionary, texts, num_topics=20):
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    del texts
    lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
    lsi = gensim.models.lsimodel.LsiModel(corpus, num_topics=num_topics, id2word=dictionary)
    return lda, lsi


def char_ngrams(n, word):
    return [word[i:i + n] for i in range(len(word)-n+1)]


def prepare_df(path):
    df = pd.read_csv(path)
    df = df.fillna(' ')

    # 斷詞(中文的話這段要另外做斷詞)
    df['q1_split'] = df['question1'].map(lambda x: str(x).lower().split())
    df['q2_split'] = df['question2'].map(lambda x: str(x).lower().split())
    return df


def load_w2v(path):
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    norm_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    norm_model.init_sims(replace=True)
    return model, norm_model


def build_features(data, stops):
    X = pd.DataFrame()

    log.info('Calculate tfidf')
    qs = pd.Series(data['question1'].tolist() + data['question2'].tolist())
    tf_st = time.time()
    weights = calculate_tfidf(qs)
    log.info('...time for cal tfidf: %.2f m' % ((time.time()-tf_st) / 60))
    del qs

    log.info('Building features')
    X['len_q1'] = data.question1.apply(word_len)   # 1:Length of Q1 str
    X['len_q2'] = data.question2.apply(word_len)   # 2:Length of Q2 str
    X['len_diff'] = abs(X.len_q1 - X.len_q2)   # 3:Length difference between Q1 and Q2

    log.info('Building char features')
    X['len_char_q1'] = data.q1_split.apply(word_len_char)  # 4:Char length of Q1
    X['len_char_q2'] = data.q2_split.apply(word_len_char)  # 5:Char length of Q2
    X['len_char_diff'] = data.apply(len_char_diff, axis=1, raw=True)  # 6:Char length difference between Q1 and Q2
    X['char_diff_unq_stop'] = data.apply(char_diff_unique_stop, stops=stops, axis=1, raw=True)  # 7: set(6)
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)  # 8:Char length Q1 / char length Q2

    log.info('Building word count features')
    X['word_count_q1'] = data.q1_split.apply(word_count)  # 9:Word count of Q1
    X['word_count_q2'] = data.q2_split.apply(word_count)  # 10:Word count of Q2
    X['word_count_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 11:Word count difference between  Q1 and Q2
    X['word_count_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 12:Word count Q1 / word count Q2

    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 13:Word count set(Q1 + Q2)
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 14:Word count set(Q1) - word count set(Q2)
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 15:Word count set(Q1) / word count set(Q2)

    X['total_unq_words_stop'] = data.apply(total_unq_words_stop, stops=stops, axis=1, raw=True)  # 16: 13 - stop words
    X['wc_diff_unique_stop'] = data.apply(wc_diff_unique_stop, stops=stops, axis=1, raw=True)  # 17: 14 - stop words
    X['wc_ratio_unique_stop'] = data.apply(wc_ratio_unique_stop, stops=stops, axis=1, raw=True)  # 18: 15 - stop words

    log.info('Building mark features')
    X['same_start'] = data.apply(same_start_word, axis=1, raw=True)  # 19 same start = 1 else = 0
    X['same_end'] = data.apply(same_end_word, axis=1, raw=True)  # 20 same end = 1 else = 0

    X['num_capital_q1'] = data.question1.apply(num_capital)  # 21
    X['num_capital_q2'] = data.question2.apply(num_capital)  # 22
    X['num_capital_diff'] = abs(X.num_capital_q1 - X.num_capital_q2)  # 23

    X['num_ques_mark_q1'] = data.question1.apply(num_ques_mark)  # 24
    X['num_ques_mark_q2'] = data.question2.apply(num_ques_mark)  # 25
    X['num_ques_mark_diff'] = abs(X.num_ques_mark_q1 - X.num_ques_mark_q2)  # 26

    log.info('Building another features')
    # 27 ~ 27+28(14*2)-1=54: First word in sentence(one hot)
    for start in common_start_words:
        X['start_%s_%s' % (start, 'q1')] = data.q1_split.apply(start_with, args=(start,))
    for start in common_start_words:  # 為了讓csv看起來更漂亮(更像one hot)
        X['start_%s_%s' % (start, 'q2')] = data.q2_split.apply(start_with, args=(start,))

    X['common_words'] = data.apply(common_words, axis=1, raw=True)  # 55:兩句相同的字數
    X['common_words_unique'] = data.apply(common_words_unit, axis=1, raw=True)  # 56:兩句相同的字母數

    X['word_match'] = data.apply(word_match_share, axis=1, raw=True)  # 57:字的重複比例 between Q1 and Q2
    X['word_match_stops'] = data.apply(word_match_share_stops, stops=stops,
                                       axis=1, raw=True)  # 58:字的重複比例 without stop word between Q1 and Q2
    X['tfidf_wm'] = data.apply(tfidf_word_match_share, weights=weights,
                               axis=1, raw=True)  # 59:字的重複比例 between Q1 and Q2 (TF-IDF值)
    X['tfidf_wm_stops'] = data.apply(tfidf_word_match_share_stops, stops=stops, weights=weights,
                                     axis=1, raw=True)  # 60:字的重複比例 without stop word between Q1 and Q2 (TF-IDF值)

    log.info('Building fuzzy features')
    # 61~67:Build fuzzy features
    X['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
                                         axis=1)
    X['fuzz_partial_token_set_ratio'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_partial_token_sort_ratio'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                           axis=1)
    X['fuzz_token_sort_ratio'] = data.apply(
        lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 68:jaccard distance

    log.info('Build word2vec/glove distance features')
    # Build word2vec/glove distance features
    X['wmd'] = data.apply(lambda x: wmd(x['q1_split'], x['q2_split']), axis=1)  # 69
    X['norm_wmd'] = data.apply(lambda x: norm_wmd(x['q1_split'], x['q2_split']), axis=1)  # 70

    question1_vectors = np.zeros((data.shape[0], 300))

    log.info('Sent2Vec')
    # Sent2Vec
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)

    log.info('Building distance features')
    # Build distance features
    X['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                           np.nan_to_num(question2_vectors))]
    X['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                 np.nan_to_num(question2_vectors))]
    X['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                               np.nan_to_num(question2_vectors))]
    X['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]
    X['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

    X['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    X['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    X['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    X['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]  # 79

    return X


def build_topic_feats(data):
    X = pd.DataFrame()

    # Cosine similarity
    def _lda_sim(row):
        return gensim.matutils.cossim(_lda(row['question1']), _lda(row['question2']))

    def _lsi_sim(row):
        return gensim.matutils.cossim(_lsi(row['question1']), _lsi(row['question2']))

    # Hellinger similarity
    def _lda_hellinger_sim(row):
        dense1 = gensim.matutils.sparse2full(_lda(row['question1']), num_topics)
        dense2 = gensim.matutils.sparse2full(_lda(row['question2']), num_topics)
        return np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())

    def _lsi_hellinger_sim(row):
        dense1 = gensim.matutils.sparse2full(_lsi(row['question1']), num_topics)
        dense2 = gensim.matutils.sparse2full(_lsi(row['question2']), num_topics)
        return np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())

    # Get a list about LDA topic probability like [(topic N, probability),...]. ex.[(10, 0.013), (12, 0.431)]
    def _lda(x):
        return lda_model[dictionary.doc2bow(clean_doc(str(x)))]

    # Get a list about LSI topic probability like [(topic N, probability),...]. ex.[(10, 0.013), (12, 0.431)]
    def _lsi(x):
        return lsi_model[dictionary.doc2bow(clean_doc(str(x)))]

    # LDA features
    X['lda_cos_sim'] = data.apply(_lda_sim, axis=1, raw=True)
    X['lda_hellinger_sim'] = data.apply(_lda_hellinger_sim, axis=1, raw=True)

    topics_q1 = data.question1.apply(lambda x: dict(lda_model[dictionary.doc2bow(clean_doc(x))]))
    for idx in range(num_topics):
        X['lda_topic_%s_%s' % (idx, 'q1')] = topics_q1.apply(lambda x: x.get(idx, 0))
    del topics_q1
    topics_q2 = data.question2.apply(lambda x: dict(lda_model[dictionary.doc2bow(clean_doc(x))]))
    for idx in range(num_topics):
        X['lda_topic_%s_%s' % (idx, 'q2')] = topics_q2.apply(lambda x: x.get(idx, 0))
    del topics_q2

    # LSI features
    X['lsi_cos_sim'] = data.apply(_lsi_sim, axis=1, raw=True)
    X['lsi_hellinger_sim'] = data.apply(_lsi_hellinger_sim, axis=1, raw=True)

    topics_q1 = data.question1.apply(lambda x: dict(lsi_model[dictionary.doc2bow(clean_doc(x))]))
    for idx in range(num_topics):
        X['lsi_topic_%s_%s' % (idx, 'q1')] = topics_q1.apply(lambda x: x.get(idx, 0))
    del topics_q1
    topics_q2 = data.question2.apply(lambda x: dict(lsi_model[dictionary.doc2bow(clean_doc(x))]))
    for idx in range(num_topics):
        X['lsi_topic_%s_%s' % (idx, 'q2')] = topics_q2.apply(lambda x: x.get(idx, 0))
    del topics_q2

    return X


if __name__ == '__main__':
    test = True
    # test = False

    common_feats = False
    topic_model = True

    log = set_logger()

    process_time = []
    save_path = []

    if test:
        data_path = '/data1/quora_pair/50q_pair.csv'
        data_path_test = '/data1/quora_pair/50q_pair_test.csv'
        w2v_path = '/data1/resources/GoogleNews-vectors-negative300.bin'
        out_path = 'added_features/'
        num_topics = 20
    else:
        data_path = '/home/csist/Dataset/QuoraQP/train_clean.csv'
        data_path_test = '/home/csist/Dataset/QuoraQP/test_clean.csv'
        w2v_path = '/home/csist/workspace/resources/GoogleNews-vectors-negative300.bin'
        out_path = 'added_features/'
        num_topics = 50
    log.info('stop words: {0}'.format(stop_words))

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # Read data frame and build split feature for instance '1 2 3' to ['1', '2', '3']
    log.info('Reading data frame')
    st = time.time()
    df = prepare_df(data_path)
    rd_time = time.time()-st

    # Read test data
    log.info('Reading test data frame')
    st = time.time()
    dft = prepare_df(data_path_test)
    rdt_time = time.time()-st

    process_time.append('...time for read data frame: %.2f s' % rd_time)
    process_time.append('...time for read test data: %.2f s' % rdt_time)

    # Build common features if common_feats = True
    if common_feats:
        meta = {
            'train': {'df': df,
                      'out': out_path+'train_features.csv'},
            'test': {'df': dft,
                     'out': out_path+'test_features.csv'}
        }
        # Load word vector model
        log.info('Loading w2v')
        st = time.time()
        model, norm_model = load_w2v(w2v_path)
        # model, norm_model = None, None
        w2v_time = time.time()-st
        process_time.append('...time for load w2v: %.2f m' % (w2v_time / 60))

        for task_name, task in meta.items():
            data = task.get('df')
            path = task.get('out')

            # Build features
            log.info('Building features')
            st = time.time()
            df_new_feature = build_features(data, stop_words)
            build_time = time.time()-st

            # Save feature data to csv
            save_path.append('save %s feats csv in %s' % (task_name, path))
            st = time.time()
            df_new_feature.to_csv(path, index=False)
            save_time = time.time()-st

            process_time.append('...time for build %s features: %.2f m' % (task_name, build_time / 60))
            process_time.append('...time for save %s csv: %.2f m' % (task_name, save_time / 60))

    # Build topic model features if topic_model = True
    if topic_model:
        meta = {
            'train': {'df': df,
                      'out': out_path + '%d_train_topic_feats.csv' % num_topics},
            'test': {'df': dft,
                     'out': out_path + '%d_test_topic_feats.csv' % num_topics}
        }

        model_out_path = out_path + 'models/'
        lda_model_path = model_out_path + '%d_quora_lda.model' % num_topics
        lsi_model_path = model_out_path + '%d_quora_lsi.model' % num_topics
        dictionary_path = model_out_path + 'dictionary.pickle'

        if not os.path.isdir(model_out_path):
            os.makedirs(model_out_path)

        if not os.path.isfile(lda_model_path):
            # Build LDA model
            log.info('Building LDA and LSI model')
            log.warning('**************** dictionary should be build by all data... ********************')
            st = time.time()
            texts = [i for i in pd.Series(df['question1'].tolist() + df['question2'].tolist()).apply(clean_doc)]
            # If cant find dictionary.pickle re-train a dictionary
            if not os.path.isfile(dictionary_path):
                dictionary = gensim.corpora.Dictionary(texts)
                # Saving dictionary to pickle file
                with open(dictionary_path, 'wb') as handle:
                    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(dictionary_path, 'rb') as handle:
                    dictionary = pickle.load(handle)

            lda_model, lsi_model = train_lda(dictionary, texts, num_topics=num_topics)
            lda_time = time.time()-st
            del texts
            process_time.append('...time for train lda and lsi: %.2f m' % (lda_time / 60))

            # Saving lda_model, lsi_model to model file
            lda_model.save(lda_model_path)
            lda_model.save(lsi_model_path)

        with open(dictionary_path, 'rb') as handle:
            dictionary = pickle.load(handle)
        lda_model = gensim.models.ldamodel.LdaModel.load(lda_model_path)
        lsi_model = gensim.models.lsimodel.LsiModel.load(lsi_model_path)

        for task_name, task in meta.items():
            data = task.get('df')
            path = task.get('out')

            # Build features
            log.info('Building features')
            st = time.time()
            df_new_feature = build_topic_feats(data)
            build_time = time.time()-st

            # Save feature data to csv
            save_path.append('save topic model %s csv in %s' % (task_name, path))
            st = time.time()
            df_new_feature.to_csv(path, index=False)
            save_time = time.time()-st

            process_time.append('...time for build %s LDA and LSI features: %.2f m' % (task_name, build_time / 60))
            process_time.append('...time for save %s LDA and LSI csv: %.2f m' % (task_name, save_time / 60))

    # Print all process time
    for p_time_info in process_time:
        log.info(p_time_info)

    for save_info in save_path:
        log.info(save_info)
