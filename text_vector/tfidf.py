import jieba
import numpy as np

# 语料
corpus = [
    '深度学习是机器学习的分支',
    '这是机器学习的简介',
    '这是自然语言处理的介绍'
]

seg_corpus = [' '.join(jieba.lcut(x)) for x in corpus]  # 对每一个句子进行分词

# 词袋，用来存放所有的词语
words_list = []
for i in range(len(seg_corpus)):
    words_list.append(seg_corpus[i].split())

print(words_list)


# 统计词语数量
from collections import Counter

# 词频统计
count_list = list()

# 遍历语料
for i in range(len(words_list)):
    # 统计词频
    count = Counter(words_list[i])
    # 词频列表
    count_list.append(count)
print(count_list)


# TF-IDF相关函数定义

def tf(word, count):
    return count[word] / sum(count.values())


def idf(word, count_list):
    n_contain = sum([1 for count in count_list if word in count])
    return np.log(len(count_list) / (1 + n_contain))


def tf_idf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

# 输出结果

for i, count in enumerate(count_list):
    print("第 {} 个文档 TF-IDF 统计信息".format(i + 1))
    scores = {word : tf_idf(word, count, count_list) for word in count}
    sorted_word = sorted(scores.items(), key = lambda x : x[1], reverse=True)
    for word, score in sorted_word:
        print("\tword: {}, TF-IDF: {}".format(word, round(score, 5)))


from gensim import corpora
# 赋给语料库中每个词(不重复的词)一个整数id
dic = corpora.Dictionary(words_list)
new_corpus = [dic.doc2bow(words) for words in words_list]
# 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
print(new_corpus)


# # 训练模型并保存
# from gensim import models
# tfidf = models.TfidfModel(new_corpus)
# tfidf.save("tfidf.model")
# # 载入模型
# tfidf = models.TfidfModel.load("tfidf.model")
# # 使用这个训练好的模型得到单词的tfidf值
# tfidf_vec = []
# for i in range(len(seg_corpus)):
#     string = seg_corpus[i]
#     string_bow = dic.doc2bow(string.lower().split())
#     string_tfidf = tfidf[string_bow]
#     tfidf_vec.append(string_tfidf)

# # 输出 词语id与词语tfidf值
# print(tfidf_vec)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorize = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_trans = TfidfTransformer(norm=None, smooth_idf=False)
tfidf = tfidf_trans.fit_transform(vectorize.fit_transform(seg_corpus))
# print(tfidf)
# seg_corpus_weight = tfidf.toarray()
# print(seg_corpus_weight)

