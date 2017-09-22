#
# Import packages
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import make_npy_files


# test = True if run on PC ; False if run on Server
test = False
#
make_npy = True
# 0 if you do not want to make cv data npy
k_fold = 0
# True if want to make word embedding npy file
fast_text = True
glove = True
w2v = True
#
# Set directories and parameters
# ----------------------------------------------------------------------------
if test:
    resource = '/data1/resources/'
    train = '/data1/quora_pair/50q_pair.csv'
    test = '/data1/quora_pair/50q_pair_test.csv'
    path_feats_train = 'added_features/train_feats.csv'
    path_feats_test = 'added_features/test_feats.csv'
else:
    resource = '/home/csist/workspace/resources/'
    train = '/home/csist/Dataset/QuoraQP/train_clean.csv'
    test = '/home/csist/Dataset/QuoraQP/test_clean.csv'
    path_feats_train = 'added_features/train_features.csv'
    path_feats_test = 'added_features/test_features.csv'

emb_npy_path = 'data/emb_npy/'
# EMB_NPY = emb_npy_path + 'glove840B300d.npy'
# EMB_NPY = emb_npy_path + 'fasttext_matrix.npy'
data_npy_path = 'data/'
TRAIN1_NPY = data_npy_path + 'train1.npy'
TRAIN2_NPY = data_npy_path + 'train2.npy'
TRAIN_LABLE = data_npy_path + 'train_label.npy'
TEST1_NPY = data_npy_path + 'test1.npy'
TEST2_NPY = data_npy_path + 'test2.npy'
TEST_ID = data_npy_path + 'test_id.npy'
tokenizer_path = data_npy_path + 'tokenizer/'

#
# Doing pre process if npy files not find
# ----------------------------------------------------------------
if make_npy:
    make_npy_files.run_data_tokenizer(train_raw=train, test_raw=test, data_npy_path=data_npy_path)

if fast_text or glove or w2v:
    make_npy_files.run_emb(data_npy_path=data_npy_path, emb_raw_path=resource, emb_npy_path=emb_npy_path,
                           fast_text=True, glove=True, w2v=True)

if k_fold:
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True)

    #
    # Load training data and testing data
    # ----------------------------------------------------------------
    data_1 = np.load(TRAIN1_NPY)
    data_2 = np.load(TRAIN2_NPY)
    labels = np.load(TRAIN_LABLE)
    print('Shape of data tensor:', data_1.shape)
    print('Shape of label tensor:', labels.shape)

    test_data_1 = np.load(TEST1_NPY)
    test_data_2 = np.load(TEST2_NPY)
    test_ids = np.load(TEST_ID)
    print('Shape of test data tensor:', test_data_1.shape)

    features = pd.read_csv(path_feats_train)
    features = ((features - features.min()) / (features.max() - features.min())).values
    print('Shape of features tensor:', features.shape)

    test_feats = pd.read_csv(path_feats_test)
    test_feats = ((test_feats - test_feats.min()) / (test_feats.max() - test_feats.min())).values
    print('Shape of test features tensor:', test_feats.shape)

    #
    # Split data in k-fold and save them to npy file
    # ----------------------------------------------------------------
    for tid, vid in kf.split(data_1, labels):
        data_1_train, data_2_train, labels_train = data_1[tid], data_2[tid], labels[tid]
        data_1_val, data_2_val, labels_val = data_1[vid], data_2[vid], labels[vid]
        feats, feats_val = features[tid], features[vid]
