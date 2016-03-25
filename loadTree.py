import json
import cPickle as pickle

import nltk.tree

import loadFile
from loadFile import aspect_label
from loadFile import rating_label
from loadFile import pair_label

UNK = "UNK"


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.left = None
        self.right = None
        self.isLeaf = False
        self.word = None
        self.hActs1 = None

    def __str__(self):
        if self.isLeaf:
            return str(self.word)
        else:
            return 'NODE ' + '(' + str(self.left) + ' ' + str(self.right) + ')'


class TreeSingleLabel:
    def __init__(self, tree, label, label_method):
        self.root = build_tree(nltk.tree.Tree.fromstring(tree))
        self.label = label_method(label)
        self.probs = None
        self.depth = tree_depth(self.root)

    def __str__(self):
        return str(self.label) + ' (' + str(self.root) + ')'


def build_tree(tree, parent=None):
    if len(tree) == 1:
        root = Node(parent)
        root.isLeaf = True
        root.word = tree.leaves()[0]
    else:
        root = Node(parent)
        root.isLeaf = False
        root.left = build_tree(tree[0], root)
        root.right = build_tree(tree[1], root)
    return root


def tree_depth(root):
    if root.isLeaf:
        return 1
    else:
        return max(tree_depth(root.left), tree_depth(root.right)) + 1


def traverse_tree(root, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right.
    Calls nodeFn at each node
    """
    nodeFn(root, args)
    if root.left is not None:
        traverse_tree(root.left, nodeFn, args)
    if root.right is not None:
        traverse_tree(root.right, nodeFn, args)


def add_word(node, words):
    if node.isLeaf:
        words.add(node.word)


def map_words(node, word_map):
    if node.isLeaf:
        if node.word not in word_map:
            node.word = word_map[UNK]
        else:
            node.word = word_map[node.word]

# 根据训练数据 得到所有树的叶子节点单词构成一个单词映射
def build_word_map(trees):
    print "Counting words to give each word an index.."

    words = set()
    for tree in trees:
        traverse_tree(tree.root, nodeFn=add_word, args=words)
    words = list(words)
    words.append(UNK)  # Add unknown as word

    word_map = dict(zip(words, xrange(len(words))))

    print "Saving wordMap to wordMap.bin"
    with open('data/wordMap.bin', 'w') as fid:
        pickle.dump(word_map, fid)
    with open('data/wordList.bin', 'w') as fid:
        pickle.dump(words, fid)
    return word_map


def load_word_map():
    with open('data/wordMap.bin', 'r') as fid:
        return pickle.load(fid)


def load_word_list():
    with open('data/wordList.bin', 'r') as fid:
        return pickle.load(fid)


def load_trees(filename, label_method):
    trees = []
    with open(filename) as handle:
        for line in handle:
            data = json.loads(line.rstrip())
            for label in data['label']:
                trees.append(TreeSingleLabel(data['tree'], label, label_method))
    return trees

# 转换成树格式
def convert_trees(trees, word_map):
    for tree in trees:
        traverse_tree(tree.root, nodeFn=map_words, args=word_map)
    return trees

def load_all(data_set, label_method):
    filename = './data/' + data_set + '.json'
    training_word_map = load_word_map()
    trees = load_trees(filename, label_method)
	
    convert_trees(trees, training_word_map)
    return trees

if __name__ == '__main__':
    train = load_trees('./data/train.json', rating_label)
    try:
        training_word_map = load_word_map()
    except IOError:
        training_word_map = build_word_map(train)
    convert_trees(train, training_word_map)
