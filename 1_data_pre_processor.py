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
