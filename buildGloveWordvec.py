import loadTree
import numpy as np
import cPickle as pickle

training_word_map = loadTree.load_word_map()
training_word_lst = loadTree.load_word_list()

glove_dic = 'data/glove.42B.300d.bin'
glove_vec = 'data/glove.42B.300d.npy'

with open(glove_dic, 'r') as fid:
    glove_word_map = pickle.load(fid)
glove_word_vec = np.load(glove_vec)

L = np.empty((300, len(training_word_lst)))
for word in training_word_lst:
    if word in glove_word_vec:
        L[:, training_word_map[word]] = glove_word_vec[glove_word_map[word]]
