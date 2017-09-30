import csv
import os
import codecs
import pickle
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import preprocessing
from gensim.models import KeyedVectors


def run_data_tokenizer(train_raw, test_raw, data_npy_path):
    #
    # Set directories and parameters
    # ----------------------------------------------------------------------------

    train_data_file = train_raw
    test_data_file = test_raw

    train1_npy = data_npy_path + 'train1.npy'
    train2_npy = data_npy_path + 'train2.npy'
    train_label = data_npy_path + 'train_label.npy'
    test1_npy = data_npy_path + 'test1.npy'
    test2_npy = data_npy_path + 'test2.npy'
    test_id = data_npy_path + 'test_id.npy'

    max_sequence_length = 30
    max_nb_words = 200000

    #
    # The function "text_to_wordlist" is from
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    # ----------------------------------------------------------------------------
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().split()

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Clean the text
        text = preprocessing.word_patterns_replace(text)

        # Optionally, shorten words to their stems
        # Ex. >>> print(stemmer.stem("running"))
        #     run
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return text

    #
    # Read Training data
    # ----------------------------------------------------------------------------
    texts_1 = []
    texts_2 = []
    labels = []
    with codecs.open(train_data_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            texts_1.append(text_to_wordlist(values[3]))
            texts_2.append(text_to_wordlist(values[4]))
            labels.append(int(values[5]))
    print('Found %s texts in train.csv' % len(texts_1))


    #
    # Read Testing data
    # ----------------------------------------------------------------------------
    test_texts_1 = []
    test_texts_2 = []
    test_ids = []
    with codecs.open(test_data_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            test_texts_1.append(text_to_wordlist(values[1]))
            test_texts_2.append(text_to_wordlist(values[2]))
            test_ids.append(values[0])
    print('Found %s texts in test.csv' % len(test_texts_1))


    #
    # Vectorize texts and turn texts into sequences
    # (=list of word indexes, where the word of rank i in the dataset (starting at 1) has index i).
    # Ex. ['i am pig', 'you are queen'] ===> [ [1, 2, 3], [4, 5, 6] ]
    #     word_index = {'i':1, 'am':2, 'pig':3, 'you':4, 'are':5, 'queen':6}
    # ----------------------------------------------------------------------------
    tokenizer_path = data_npy_path + 'tokenizer/'
    # If tokenizer not found
    if not os.path.isfile(tokenizer_path + 'tokenizer.pickle'):
        # Make dirs
        os.makedirs(tokenizer_path)
        # Training tokenizer
        tokenizer = Tokenizer(num_words=max_nb_words)
        tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
        # Saving to pickle file
        with open(tokenizer_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load tokenizer
    with open(tokenizer_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # Tokenize text data
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    # Padding sequence training data and saving to npy file
    data_1 = pad_sequences(sequences_1, maxlen=max_sequence_length)
    data_2 = pad_sequences(sequences_2, maxlen=max_sequence_length)
    labels = np.array(labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    np.save(train1_npy, data_1)
    print('train1_npy store in: %s' % train1_npy)
    np.save(train2_npy, data_2)
    print('train2_npy store in: %s' % train2_npy)
    np.save(train_label, labels)
    print('train_label store in: %s' % train_label)

    # Padding sequence testing data and saving to npy file
    test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)
    test_ids = np.array(test_ids)

    np.save(test1_npy, test_data_1)
    print('test1_npy store in: %s' % test1_npy)
    np.save(test2_npy, test_data_2)
    print('test2_npy store in: %s' % test2_npy)
    np.save(test_id, test_ids)
    print('test_ids store in: %s' % test_id)
    

def run_emb(data_npy_path, emb_raw_path, emb_npy_path, fast_text=False, glove=False, w2v=False):
    max_nb_words = 200000
    embedding_dim = 300
    tokenizer_path = data_npy_path + 'tokenizer/'

    # loading
    with open(tokenizer_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    if fast_text:
        task_name = 'fast text'
        emb_file = emb_raw_path + 'wiki.en.vec'
        emb_npy_saveto = emb_npy_path + 'fasttext_matrix.npy'
        _emb(tokenizer, task_name, emb_file, emb_npy_saveto, max_nb_words, embedding_dim)
        
    if glove:
        task_name = 'glove'
        emb_file = emb_raw_path + 'glove.840B.300d.txt'
        emb_npy_saveto = emb_npy_path + 'glove840B300d.npy'
        _emb(tokenizer, task_name, emb_file, emb_npy_saveto, max_nb_words, embedding_dim)

    if w2v:
        task_name = 'w2v'
        emb_file = emb_raw_path + 'GoogleNews-vectors-negative300.bin'
        emb_npy_saveto = emb_npy_path + 'w2v.npy'
        _emb(tokenizer, task_name, emb_file, emb_npy_saveto, max_nb_words, embedding_dim)
        print('w2v to npy file does not work')
        pass


def _emb(tokenizer, task_name, embedding_file, emb_npy_saveto, max_nb_words=200000, embedding_dim=300):
    #
    # Index word vectors
    # ----------------------------------------------------------------------------
    print('Indexing word vectors')

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))
    nb_words = min(max_nb_words, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, embedding_dim))

    if task_name == 'w2v':
        word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        print('Found %s word vectors of word2vec' % len(word2vec.vocab))
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
    else:
        embeddings_index = {}
        ef = open(embedding_file)
        count = 0
        for line in ef:
            if task_name == 'fast text' and count == 0:
                count += 1
                continue
            values = line.strip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        ef.close()
    
        print('Found %d word vectors of %s.' % (len(embeddings_index), task_name))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    np.save(emb_npy_saveto, embedding_matrix)
    print('%s npy store in: %s' % (task_name, emb_npy_saveto))
