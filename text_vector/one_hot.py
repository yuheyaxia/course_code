import numpy as np
import jieba

# 语料
corpus = [
    '深度学习是机器学习的分支',
    '这是机器学习的简介',
    '这是自然语言处理的介绍'
]

# print(f"语料的长度是：{len(corpus)}")
# print(f"第一条数据是：{corpus[0]}")

seg_corpus = [' '.join(jieba.lcut(x)) for x in corpus]  # 对每一个句子进行分词
# print(seg_corpus)
# print(len(seg_corpus))

# 手动实现onehot代码

# 词袋，用来存放所有的词语
words = []
for _ in seg_corpus:
    words.extend(_.split())
# print(words)
# print(f"词袋的原始长度是：{len(words)}")

# 对词袋进行去重
word_list = sorted(list(set(words)))
print(word_list)
# print(f"去重后的词袋的长度是：{len(word_list)}")


# 词袋中的词语转换为字典形式
word_dict= {word:index for index,word in enumerate(word_list)}
# 词典的大小
vocab_size=len(word_dict)
# print(f"词典为：{word_dict}")
# print(f"词典的大小:{vocab_size}")


def get_one_hot(index):
    """
    获得one-hot编码
    """
    # 初始化全0列表，长度为vocab_size
    one_hot=[0 for i in range(vocab_size)]
    # 词语标记对应位置设为1
    one_hot[index] = 1
    # 将列表转换成矩阵
    return np.array(one_hot)

# 将index=1的位置设为1
# get_one_hot(1)


# 将语料的第一个分词后的句子转为one-hot编码
""""
step 1: 句子获取
step 2: 转换为索引
step 3: 变为one-hot
"""
# 对第一个句子进行转换
indexs = [word_dict[i] for i in seg_corpus[0].split()]
one_hot_list = np.array([get_one_hot(index) for index in indexs])
print(one_hot_list)



# sklearn实现

from sklearn.preprocessing import LabelBinarizer
# 初始化编码器
lb = LabelBinarizer()
lb.fit(word_list)
print(lb.classes_)
sentence = seg_corpus[0].split()
# 编码
encode_sentence = lb.transform(sentence)
print(encode_sentence)
print((encode_sentence == one_hot_list).all())

# 解码
ori_seg_sent = lb.inverse_transform(encode_sentence)
print(ori_seg_sent)
