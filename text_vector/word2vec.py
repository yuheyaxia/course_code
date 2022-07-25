import jieba
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec


def get_sentence(data_file):
    # 读取数据集中的句子
    f=open(data_file, 'r', encoding='utf-8')
    reader = f.readlines()
    sentence=[]
    for line in reader:
        text, label = line.split('\t')
        sentence.append(text.strip())
    return sentence

# 读取句子语料
train_sentence=get_sentence('data/train.txt')
test_sentence=get_sentence('data/test.txt')
dev_sentence=get_sentence('data/dev.txt')

# 使用所有语料作为词向量训练语料
train_data = train_sentence + test_sentence + dev_sentence

# 分词处理
train_data=[list(jieba.cut(stentence)) for stentence in train_data]


"""
Word2vec参数：
min_count: 在不同大小的语料集中，我们对于基准词频的需求也是不一样的。譬如在较大的语料集中，我们希望忽略那些只出现过一两次的单词，这里我们就可以通过设置min_count参数进行控制。一般而言，合理的参数值会设置在0~100之间。
size: size参数主要是用来设置神经网络的层数，Word2Vec 中的默认值是设置为100层。更大的层次设置意味着更多的输入数据，不过也能提升整体的准确度，合理的设置范围为 10~数百。

"""


# sg : {0, 1}, optional Training algorithm: 1 for skip-gram; otherwise CBOW.
model = word2vec.Word2Vec(train_data, sg=1, workers=8, min_count=5, vector_size=300)

# # 保存模型
save_model_path='text_vector/word2vec.model'
model.save(save_model_path)

# 载入保存的模型
model = word2vec.Word2Vec.load(save_model_path)

# 某一个词的词向量
print(model.wv['智能'])
print(model.wv['智能'].shape)

# 相似度计算
vec_a = model.wv['智能']
vec_b = model.wv['智能机']

from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))
print(cos_sim[0][0])

# 查找最近最相似的词
print(model.wv.most_similar(['智能'],topn=10))

