import csv
import codecs

import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import preprocessing

import os


def run(emb_raw_path, emb_npy_path, data_path, train_raw, test_raw, fast_text=False, glove=False):
    #
    # Set directories and parameters
    # ----------------------------------------------------------------------------

    TRAIN_DATA_FILE = train_raw
    TEST_DATA_FILE = test_raw

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    TRAIN1_NPY = data_path + 'train1.npy'
    TRAIN2_NPY = data_path + 'train2.npy'
    TRAIN_LABLE = data_path + 'train_label.npy'
    TEST1_NPY = data_path + 'test1.npy'
    TEST2_NPY = data_path + 'test2.npy'
    TEST_LABLE = data_path + 'test_label.npy'

    MAX_SEQUENCE_LENGTH = 30
    MAX_NB_WORDS = 200000
    EMBEDDING_DIM = 300


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
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
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
    with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
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
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens' % len(word_index))

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    labels = np.array(labels)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    np.save(TRAIN1_NPY, data_1)
    print('TRAIN1_NPY store in: %s' % TRAIN1_NPY)
    np.save(TRAIN2_NPY, data_2)
    print('TRAIN2_NPY store in: %s' % TRAIN2_NPY)
    np.save(TRAIN_LABLE, labels)
    print('TRAIN_LABLE store in: %s' % TRAIN_LABLE)

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    test_ids = np.array(test_ids)

    np.save(TEST1_NPY, test_data_1)
    print('TEST1_NPY store in: %s' % TEST1_NPY)
    np.save(TEST2_NPY, test_data_2)
    print('TEST2_NPY store in: %s' % TEST2_NPY)
    np.save(TEST_LABLE, test_ids)
    print('TEST_LABLE store in: %s' % TEST_LABLE)
    
    #
    # Index word vectors
    # ----------------------------------------------------------------------------
    print('Indexing word vectors')

    # word2vec = KeyedVectors.load_word2vec_format(emb_file, binary=True)
    # print('Found %s word vectors of word2vec' % len(word2vec.vocab))

    def emb(task_name, embedding_file, emb_npy_saveto):
        embeddings_index = {}
        ef = open(embedding_file)
        count = 0
        for line in ef:
            if fast_text and count == 0:
                count += 1
                continue
            values = line.strip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        ef.close()

        print('Found %d word vectors of %s.' % (len(embeddings_index), task_name))

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        nb_words = min(MAX_NB_WORDS, len(word_index)) + 1

        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        np.save(emb_npy_saveto, embedding_matrix)
        print('%s npy store in: %s' % (task_name, emb_npy_saveto))

    if not os.path.isdir(emb_npy_path):
        os.makedirs(emb_npy_path)

    if fast_text:
        task_name = 'fast text'
        emb_file = emb_raw_path + 'wiki.en.vec'
        emb_npy_saveto = emb_npy_path + 'fasttext_matrix.npy'
        emb(task_name, emb_file, emb_npy_saveto)
        
    if glove:
        task_name = 'glove'
        emb_file = emb_raw_path + 'glove.840B.300d.txt'
        emb_npy_saveto = emb_npy_path + 'glove840B300d.npy'
        emb(task_name, emb_file, emb_npy_saveto)
