import optparse
import cPickle as pickle

import sgd as optimizer
from rntn import RNTN
from rnn import RNN
from treeLSTM import TreeLSTM
from treeTLSTM import TreeTLSTM
import time
import matplotlib.pyplot as plt
import numpy as np
import loadTree as tree
from sklearn import metrics
import pdb


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test", action="store_true", dest="test", default=False)

    # Optimizer
    parser.add_option("--minibatch", dest="minibatch", type="int", default=30)
    parser.add_option("--optimizer", dest="optimizer", type="string", default="adagrad")
    parser.add_option("--epochs", dest="epochs", type="int", default=50)
    parser.add_option("--step", dest="step", type="float", default=5e-2)
    parser.add_option("--rho", dest="rho", type="float", default=1e-3)

    # Dimension
    parser.add_option("--wvecDim", dest="wvec_dim", type="int", default=30)
    parser.add_option("--memDim", dest="mem_dim", type="int", default=30)

    parser.add_option("--outFile", dest="out_file", type="string", default="models/test.bin")
    parser.add_option("--inFile", dest="in_file", type="string", default="models/test.bin")
    parser.add_option("--data", dest="data", type="string", default="train")

    parser.add_option("--model", dest="model", type="string", default="RNN")
    parser.add_option("--label", dest="label_method", type="string", default="rating")

    (opts, args) = parser.parse_args(args)

    evaluate_accuracy_while_training = True

    if opts.label_method == 'rating':
        label_method = tree.rating_label
        opts.output_dim = 5
    elif opts.label_method == 'aspect':
        label_method = tree.aspect_label
        opts.output_dim = 5
    elif opts.label_method == 'pair':
        label_method = tree.pair_label
        opts.output_dim = 25
    else:
        raise '%s is not a valid labelling method.' % opts.label_method

    # Testing
    if opts.test:
        test(opts.in_file, opts.data, label_method, opts.model)
        return

    print "Loading data..."
    train_accuracies = []
    dev_accuracies = []
    # load training data
    trees = tree.load_trees('./data/train.json', label_method)
    training_word_map = tree.load_word_map()
    opts.num_words = len(training_word_map)
    tree.convert_trees(trees, training_word_map)
    labels = [each.label for each in trees]
    count = np.zeros(opts.output_dim)
    for label in labels: count[label] += 1
    # weight = 10 / (count ** 0.1)
    weight = np.ones(opts.output_dim)

    if opts.model == 'RNTN':
        nn = RNTN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho)
    elif opts.model == 'RNN':
        nn = RNN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho, weight=weight)
    elif opts.model == 'TreeLSTM':
        nn = TreeLSTM(opts.wvec_dim, opts.mem_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho)
    elif opts.model == 'TreeTLSTM':
        nn = TreeTLSTM(opts.wvec_dim, opts.mem_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN' % opts.model

    nn.init_params()

    sgd = optimizer.SGD(nn, alpha=opts.step, minibatch=opts.minibatch, optimizer=opts.optimizer)

    dev_trees = tree.load_trees('./data/dev.json', label_method)
    tree.convert_trees(dev_trees, training_word_map)
	# epochs 表示交叉验证的次数
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d" % e
        sgd.run(trees, e)
        end = time.time()
        print "Time per epoch : %f" % (end - start)

        with open(opts.out_file, 'w') as fid:
            pickle.dump(opts, fid)
            pickle.dump(sgd.costt, fid)
            nn.to_file(fid)
        if evaluate_accuracy_while_training:
            # pdb.set_trace()
            print "testing on training set real quick"
            train_accuracies.append(test(opts.out_file, "train", label_method, opts.model, trees))
            print "testing on dev set real quick"
            dev_accuracies.append(test(opts.out_file, "dev", label_method, opts.model, dev_trees))

    if evaluate_accuracy_while_training:
        print train_accuracies
        print dev_accuracies
        plt.plot(train_accuracies, label='train')
        plt.plot(dev_accuracies, label='dev')
        plt.legend(loc=2)
        plt.axvline(x=np.argmax(dev_accuracies), linestyle='--')
        plt.show()


def test(net_file, data_set, label_method, model='RNN', trees=None):
    if trees is None:
        trees = tree.load_all(data_set, label_method)
    assert net_file is not None, "Must give model to test"
    print "Testing netFile %s" % net_file
    with open(net_file, 'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)

        if model == 'RNTN':
            nn = RNTN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch)
        elif model == 'RNN':
            nn = RNN(opts.wvec_dim, opts.output_dim, opts.num_words, opts.minibatch)
        elif opts.model == 'TreeLSTM':
            nn = TreeLSTM(opts.wvec_dim, opts.mem_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho)
        elif opts.model == 'TreeTLSTM':
            nn = TreeTLSTM(opts.wvec_dim, opts.mem_dim, opts.output_dim, opts.num_words, opts.minibatch, rho=opts.rho)
        else:
            raise '%s is not a valid neural network so far only RNTN, RNN, RNN2, RNN3, and DCNN' % opts.model

        nn.init_params()
        nn.from_file(fid)

    print "Testing %s..." % model

    cost, correct, guess = nn.cost_and_grad(trees, test=True)
    correct_sum = 0
    for i in xrange(0, len(correct)):
        correct_sum += (guess[i] == correct[i])

    confusion = [[0 for i in range(nn.output_dim)] for j in range(nn.output_dim)]
    for i, j in zip(correct, guess): confusion[i][j] += 1
    # makeconf(confusion)

    pre, rec, f1, support = metrics.precision_recall_fscore_support(correct, guess)
    #print "Cost %f, Acc %f" % (cost, correct_sum / float(len(correct)))
    #return correct_sum / float(len(correct))
    f1 = (100*sum(f1[1:] * support[1:])/sum(support[1:]))
    print "Cost %f, F1 %f, Acc %f" % (cost, f1, correct_sum / float(len(correct)))
    return f1


def makeconf(conf_arr):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        if a != 0:
            for j in i:
                tmp_arr.append(float(j) / float(a))
        else:
            for j in i:
                tmp_arr.append(0)
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    # plt.savefig("pathname/image.png")

    plt.show()


if __name__ == '__main__':
    run()
