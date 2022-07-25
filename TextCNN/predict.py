#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   predict.py
@Time    :   2022/07/14 11:34:36
@Author  :   M. 
@Desc    :   模型预测
'''
import jieba
import pickle
import torch
from model import Config, TextCNN


UNK, PAD = '<UNK>', '<PAD>'


def build_data(config, input_text, use_word):
    if use_word:
        tokenizer = lambda x: list(jieba.cut(x))  # word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")
    lin = input_text.strip()
    if not lin:
        raise Exception('input is null')
    content = lin
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if config.pad_size:
        if len(token) < config.pad_size:
            token.extend([PAD] * (config.pad_size - len(token)))
        else:
            token = token[:config.pad_size]
            seq_len = config.pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    ids = torch.LongTensor([words_line]).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    return ids, seq_len


def predict(config, input_text, use_word):
    data = build_data(config, input_text, use_word)
    with torch.no_grad():
        outputs = model(data)
        label = torch.argmax(outputs)
    return label.data.cpu().numpy()


if __name__ == '__main__':
    dataset = '../data'  # 存放数据集的目录
    model_name = 'TextCNN'
    embedding = 'word_embeddings.npz'

    use_word = True  # change

    config = Config(dataset, embedding)
    model = TextCNN(config).to(config.device)
   
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)

    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
    test_text = '海淀区领秀新硅谷宽景大宅预计10月底开盘'
    
    print(predict(config, test_text, use_word=use_word))