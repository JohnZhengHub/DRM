from sklearn.linear_model import SGDClassifier
import validation as vld
import loadFile
import numpy as np

reg = 0.0001


def single_train(data, label):
    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=reg)
    return clf.fit(data, label)


def train_both(data, label):
    clf1 = SGDClassifier(loss='hinge', penalty='l2', alpha=reg)
    clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=reg)
    return clf1.fit(data, label[:, 0]), clf2.fit(data, label[:, 1])

if __name__ == '__main__':
    total_data = loadFile.file2mat('./data/final_review_set.csv')
    shuffled_data = vld.data_reshuffle(total_data)
    train_mat = shuffled_data[0]
    aspect_label = shuffled_data[1]
    rating_label = shuffled_data[2]
    label_mat = np.vstack((aspect_label, rating_label)).T
    single_label = aspect_label * len(loadFile.aspect_dic) + rating_label
    print "SAS, aspect:\t", vld.cross_validation(train_mat, aspect_label, single_train, vld.test_single)
    print "SAS, rating:\t", vld.cross_validation(train_mat, rating_label, single_train, vld.test_single)
    print "SAS, both:\t", vld.cross_validation(train_mat, single_label, single_train, vld.test_single)
    print "JMAS, aspect:\t", vld.cross_validation(train_mat, single_label, single_train, vld.test_aspect)
    print "JMAS, rating:\t", vld.cross_validation(train_mat, single_label, single_train, vld.test_rating)
    print "JMAS, both:\t", vld.cross_validation(train_mat, label_mat, train_both, vld.test_mat)
