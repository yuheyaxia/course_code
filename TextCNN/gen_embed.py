#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gen_embed.py
@Time    :   2022/07/22 16:11:09
@Author  :   M. 
@Desc    :   word2vec向量转换 生成pre_trained weight
'''
import os
import jieba
import numpy as np
import pickle as pkl
from gensim.models import word2vec
from utils import build_vocab


train_dir = "data/train.txt"
vocab_dir = "data/word_vocab.pkl"
w2v_file = 'text_vector/word2vec.model'

wvmodel = word2vec.Word2Vec.load(w2v_file)

# MAX_VOCAB_SIZE = 10000
MAX_VOCAB_SIZE = len(wvmodel.wv.index_to_key) # word2vec的词的数量
emb_dim = 300

if os.path.exists(vocab_dir):
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
else:
    tokenizer = lambda x: list(jieba.cut(x))  # 以词为单位构建词表(数据集中词之间以空格隔开)
    # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    id_to_word = {v:k for k,v in word_to_id.items()}
    pkl.dump(word_to_id, open(vocab_dir, 'wb'))

embeddings = np.random.rand(len(word_to_id), emb_dim)
# print(embeddings.shape)
# print(len(wvmodel.wv.index_to_key))
for word in wvmodel.wv.index_to_key:
    if word in word_to_id:
        idx = word_to_id[word]
        embeddings[idx] = np.asarray(wvmodel.wv[word], dtype='float32')
np.savez_compressed('data/word_embeddings', embeddings=embeddings)
