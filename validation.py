import random
import loadFile
import numpy as np
from sklearn import metrics


def data_reshuffle(data_list):
    data_len = data_list[0].shape[0]
    index = range(data_len)
    random.shuffle(index)
    return [data[index] for data in data_list]


def cross_validation(data_train, data_label, train_method, validate_method, fold=10):
    instance = data_train.shape[0]
    test_num = instance / 10
    test_accuracy = 0
    train_accuracy = 0
    for fold_idx in xrange(fold):
        test_idx = set(range(fold_idx * test_num, min((fold_idx + 1) * test_num, instance)))
        train_idx = [i for i in range(instance) if i not in test_idx]
        test_idx = list(test_idx)
        model = train_method(data_train[train_idx], data_label[train_idx])
        accuracy_curr = validate_method(data_train[test_idx], data_label[test_idx], model)
        train_accuracy_curr = validate_method(data_train[train_idx], data_label[train_idx], model)
        test_accuracy += accuracy_curr
        train_accuracy += train_accuracy_curr
    test_accuracy /= fold
    train_accuracy /= fold
    return train_accuracy, test_accuracy


def test_single(data, label, model):
    prediction = model.predict(data)
    #return float(np.sum(prediction == label)) / len(label)
    pre, rec, f1, support = metrics.precision_recall_fscore_support(label, prediction)
    f1 = (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    return f1


def test_rating(data, label, model):
    prediction = model.predict(data)
    #return float(np.sum(prediction % len(loadFile.aspect_dic) == (label % len(loadFile.aspect_dic)))) / len(label)
    prediction = prediction % len(loadFile.aspect_dic)
    label = label % len(loadFile.aspect_dic)
    pre, rec, f1, support = metrics.precision_recall_fscore_support(label, prediction)
    f1 = (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    return f1


def test_aspect(data, label, model):
    prediction = model.predict(data)
    #return float(np.sum(prediction // len(loadFile.aspect_dic) == (label // len(loadFile.aspect_dic)))) / len(label)
    prediction = prediction // len(loadFile.aspect_dic)
    label = label // len(loadFile.aspect_dic)
    pre, rec, f1, support = metrics.precision_recall_fscore_support(label, prediction)
    f1 = (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    return f1


def test_mat(data, label, model):
    prediction1 = model[0].predict(data)
    prediction2 = model[1].predict(data)
    #return float(np.logical_and(prediction1 == label[:, 0], prediction2 == label[:, 1]).sum()) / len(label)
    label = label[:, 0] * 100 + label[:, 1]
    prediction = prediction1 * 100 + prediction2
    pre, rec, f1, support = metrics.precision_recall_fscore_support(label, prediction)
    f1 = (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    return f1
