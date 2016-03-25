import csv
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


aspect_dic = {"overall":    0,
              "appearance": 1,
              "taste":      2,
              "palate":     3,
              "aroma":      4}


def tokenize(data_set):
	"""
	将句子进行切分 获得单词标记
	"""
    for i in xrange(len(data_set)):
        data_set[i]['review'] = nltk.word_tokenize(data_set[i]['review'])
    return


def build_vocabulary(review_set):
    vocabulary = {}
    index = 0
    for review in review_set:
        review = set(review)
        for word in review:
            if word not in vocabulary:
                vocabulary[word] = {"index": index, "df": 1}
                index += 1
            else:
                vocabulary[word]["df"] += 1
    return vocabulary


def aspect_label(label_dic):
    return aspect_dic[label_dic['aspect']]


def rating_label(label_dic):
    # if label_dic['rating'] > 4: return 2
    # elif label_dic['rating'] == 4: return 1
    # else: return 0
    # if label_dic['rating'] > 4: return 4
    return int(label_dic['rating']) - 1


def pair_label(label_dic):
    return aspect_label(label_dic) * len(aspect_dic) + rating_label(label_dic)

# 获得每个样本的apsect 标签
def collect_aspect_label(data_set):
    label = np.zeros(len(data_set), dtype='int32')
    for i, data_entry in enumerate(data_set):
        label[i] = aspect_label(data_entry)
    return label

# 获得每个样本的rate标签
def collect_rating_label(data_set):
    label = np.zeros(len(data_set), dtype='int32')
    for i, data_entry in enumerate(data_set):
        #if data_entry['rating'] > 4:
            #label[i] = 4
        #else:
        label[i] = rating_label(data_entry)
    return label

# 加载数据 从excel表格中获取数据  表格中第一个单元是句子的单词， 第二个为 sapect关系，第三个为关系程序值 
def load(filename):
    input_file = open(filename, 'r')
    csv_reader = csv.reader(input_file)
    data_set = []
    for line in csv_reader:
        data_set.append({'review': line[0].lower(), 'aspect': line[1], 'rating': float(line[2])})
    input_file.close()
    return data_set


def file2mat(filename):
    transformer = TfidfTransformer()
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1,1))
    data = load(filename)
    reviews = [each_data['review'] for each_data in data]
    bag_of_word = vectorizer.fit_transform(reviews)
    tfidf = transformer.fit_transform(bag_of_word)

    aspect_label = collect_aspect_label(data)
    rating_label = collect_rating_label(data)
    return tfidf, aspect_label, rating_label

# 加载wordVec 的单词向量 构成一个单词向量库
def load_wordvec(filename):
    wordvec_dic = {}
    with open(filename, 'r') as wordvec_file:
        for i, line in enumerate(wordvec_file):
            content = line.split()
            wordvec_dic[content[0]] = np.array(map(float, content[1:]))
            if i % 10000 == 0:
                print i
    return wordvec_dic

# 句子中的单词向量求和
def bag_of_wordvec(tokens, wordvec_dic, dimension):
    vec = np.zeros(dimension)
    for token in tokens:
        try:
            vec += wordvec_dic[token]
        except KeyError:
            pass
            # print "No token %s" % token
    return vec

# 获取bag_of_wordvec(每个句子的所有向量相加得到的一个向量)
def file2mat_bag_of_wordvec(filename):
    dimension = 300
    wordvec_dic = load_wordvec('./data/glove.6B.300d.txt')
    data = load(filename)
    tokenize(data)
    reviews = [each_data['review'] for each_data in data]
    bag_of_wordvec_mat = np.zeros((len(reviews), dimension))
    for i in xrange(bag_of_wordvec_mat.shape[0]):
        bag_of_wordvec_mat[i] = bag_of_wordvec(reviews[i], wordvec_dic, dimension)
    aspect_label = collect_aspect_label(data)
    rating_label = collect_rating_label(data)
    return bag_of_wordvec_mat, aspect_label, rating_label

if __name__ == '__main__':
    print file2mat_bag_of_wordvec('./data/final_review_set.csv')
    pass
