#
# Import packages
# ----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import configparser
import make_npy_files


config = configparser.ConfigParser()
config.read('Config.ini')

make_npy = config.getboolean('step1', 'make_npy')

# True if want to make word embedding npy file
fast_text = config.getboolean('step1', 'fast_text')
glove = config.getboolean('step1', 'glove')
w2v = config.getboolean('step1', 'w2v')
#
# Set directories and parameters
# ----------------------------------------------------------------------------
resource = config.get('step1', 'emb_path')
train = config.get('step1', 'train_raw')
test = config.get('step1', 'test_raw')

# resource = '/home/csist/workspace/resources/'
# train = '/home/csist/Dataset/QuoraQP/train_clean.csv'
# test = '/home/csist/Dataset/QuoraQP/test_clean.csv'
# path_feats_train = 'added_features/train_features.csv'
# path_feats_test = 'added_features/test_features.csv'

data_npy_path = 'data/'
emb_npy_path = data_npy_path + 'emb_npy/'

if not os.path.isdir(data_npy_path):
    os.makedirs(data_npy_path)

if not os.path.isdir(emb_npy_path):
    os.makedirs(emb_npy_path)

#
# Doing pre process if npy files not find
# ----------------------------------------------------------------
if make_npy:
    make_npy_files.run_data_tokenizer(train_raw=train, test_raw=test, data_npy_path=data_npy_path)

if fast_text or glove or w2v:
    make_npy_files.run_emb(data_npy_path=data_npy_path, emb_raw_path=resource, emb_npy_path=emb_npy_path,
                           fast_text=True, glove=True, w2v=True)
