#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/07/14 11:26:05
@Author  :   M. 
@Desc    :   程序运行入口
'''

import time

import numpy as np
import torch

from model import Config, TextCNN
from train_eval import train, init_network
from utils import build_dataset, build_iterator, get_time_dif



dataset = '../data'  # 数据集
use_word = True  # True word-level    False char-level
# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
# embedding = 'embedding_SougouNews.npz'
embedding = 'word_embeddings.npz'

model_name = 'TextCNN'

config = Config(dataset, embedding)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样
start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = build_dataset(config, use_word)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
config.n_vocab = len(vocab)
model = TextCNN(config).to(config.device)
if model_name != 'Transformer':
    init_network(model)
print(model.parameters)
train(config, model, train_iter, dev_iter, test_iter)
