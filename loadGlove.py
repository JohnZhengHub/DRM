import numpy as np
import cPickle as pickle


def load_wordvec(filename, save_filename, save_numpyfile):
    wordvec_dic = {}
    wordvec_lst = []
    with open(filename, 'r') as wordvec_file:
        for i, line in enumerate(wordvec_file):
            content = line.split()
            wordvec_dic[content[0]] = i
            wordvec_lst.append(np.array(map(float, content[1:]), dtype='float32'))
            if 0 == i % 10000:
                print i
    wordvec_lst = np.array(wordvec_lst)
    np.save(save_numpyfile, wordvec_lst)
    with open(save_filename, 'wb') as save_file:
        pickle.dump(wordvec_dic, save_file)
    return


def load_batch(file_prefix):
    load_wordvec(file_prefix + '.txt', file_prefix + '.bin', file_prefix + '.npy')


if __name__ == '__main__':
    load_batch('./data/glove.42B.300d')
