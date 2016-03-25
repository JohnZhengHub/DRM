import numpy as np
import collections
import cPickle as pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TreeTLSTM:
    def __init__(self, wvec_dim, mem_dim, output_dim, num_words, mb_size=30, rho=1e-3, L=None):
        np.random.seed(59)
        self.wvec_dim = wvec_dim
        self.mem_dim = mem_dim
        self.output_dim = output_dim
        self.num_words = num_words
        self.mb_size = mb_size
        self.default_vec = lambda: np.zeros((wvec_dim,))
        self.rho = rho
        if L is None:
            # Word vectors
            self.L = np.random.randn(self.wvec_dim, self.num_words) * 0.01
        else:
            self.L = L.copy()

    def init_params(self):
        self.keep = 0.5

        # Input layer
        self.W_in = np.random.randn(self.mem_dim, self.wvec_dim) * 0.5
        self.b_in = np.zeros(self.mem_dim)
        self.W_out = np.random.randn(self.mem_dim, self.wvec_dim) * 0.5
        self.b_out = np.zeros(self.mem_dim)

        # Gates
        self.Ui = np.random.randn(self.mem_dim, 2 * self.mem_dim) * 0.05
        self.bi = np.ones(self.mem_dim) * 0
        self.Uf_l = np.random.randn(self.mem_dim, 2 * self.mem_dim) * 0.05
        self.Uf_r = np.random.randn(self.mem_dim, 2 * self.mem_dim) * 0.05
        self.bf = np.ones(self.mem_dim) * 0
        self.Uo = np.random.randn(self.mem_dim, 2 * self.mem_dim) * 0.05
        self.bo = np.ones(self.mem_dim) * 0
        self.Uu = np.random.randn(self.mem_dim, 2 * self.mem_dim) * 0.01
        self.bu = np.ones(self.mem_dim) * 0
        self.V = np.random.randn(self.mem_dim, 2 * self.mem_dim, 2 * self.mem_dim) * 0.1

        # Softmax weights
        self.Ws = np.random.randn(self.output_dim, self.mem_dim) * 0.01
        self.bs = np.zeros(self.output_dim)

        self.stack = [self.L, self.W_in, self.b_in,
                      self.W_out, self.b_out,
                      self.Ui, self.bi,
                      self.Uf_l, self.Uf_r, self.bf,
                      self.Uo, self.bo,
                      self.Uu, self.bu, self.V,
                      self.Ws, self.bs]

        # Gradients
        self.dW_in = np.empty(self.W_in.shape)
        self.db_in = np.empty(self.b_in.shape)
        self.dW_out = np.empty(self.W_out.shape)
        self.db_out = np.empty(self.b_out.shape)
        self.dUi = np.empty(self.Ui.shape)
        self.dbi = np.empty(self.bi.shape)
        self.dUf_l = np.empty(self.Uf_l.shape)
        self.dUf_r = np.empty(self.Uf_r.shape)
        self.dbf = np.empty(self.bf.shape)
        self.dUo = np.empty(self.Uo.shape)
        self.dbo = np.empty(self.bo.shape)
        self.dUu = np.empty(self.Uu.shape)
        self.dbu = np.empty(self.bu.shape)
        self.dV = np.empty(self.V.shape)
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty(self.bs.shape)

    def cost_and_grad(self, mbdata, test=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W, Ws, b, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total

        """
        cost = 0.0
        correct = []
        guess = []
        self.L, self.W_in, self.b_in,\
        self.W_out, self.b_out,\
        self.Ui, self.bi,\
        self.Uf_l, self.Uf_r, self.bf,\
        self.Uo, self.bo,\
        self.Uu, self.bu, self.V,\
        self.Ws, self.bs = self.stack

        # Zero gradients
        self.dW_in[:] = 0
        self.db_in[:] = 0
        self.dW_out[:] = 0
        self.db_out[:] = 0
        self.dUi[:] = 0
        self.dUf_l[:] = 0
        self.dUf_r[:] = 0
        self.dbf[:] = 0
        self.dUo[:] = 0
        self.dbo[:] = 0
        self.dUu[:] = 0
        self.dbu[:] = 0
        self.dV[:] = 0
        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.default_vec)

        # Forward prop each tree in minibatch
        for tree in mbdata:
            c, predict = self.forward_prop(tree, test)
            cost += c
            guess.append(predict)
            correct.append(tree.label)
        if test:
            return (1. / len(mbdata)) * cost, correct, guess

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.back_prop(tree)

        # scale cost and grad by mb size
        scale = (1. / self.mb_size)
        for v in self.dL.itervalues():
            v *= scale

        # Add L2 Regularization
        cost += (self.rho / 2) * np.sum(self.W_in ** 2)
        cost += (self.rho / 2) * np.sum(self.W_out ** 2)
        cost += (self.rho / 2) * np.sum(self.Ui ** 2)
        cost += (self.rho / 2) * np.sum(self.Uf_l ** 2)
        cost += (self.rho / 2) * np.sum(self.Uf_r ** 2)
        cost += (self.rho / 2) * np.sum(self.Uo ** 2)
        cost += (self.rho / 2) * np.sum(self.Uu ** 2)
        cost += (self.rho / 2) * np.sum(self.V ** 2)
        cost += (self.rho / 2) * np.sum(self.Ws ** 2)

        return scale * cost, [self.dL, scale * (self.dW_in + self.rho * self.W_in), scale * self.db_in,
                              scale * (self.dW_out + self.rho * self.W_out), scale * self.db_out,
                              scale * (self.dUi + self.rho * self.Ui), scale * self.dbi,
                              scale * (self.dUf_l + self.rho * self.Uf_l),
                              scale * (self.dUf_r + self.rho * self.Uf_r), scale * self.dbf,
                              scale * (self.dUo + self.rho * self.Uo), scale * self.dbo,
                              scale * (self.dUu + self.rho * self.Uu), scale * self.dbu,
                              scale * (self.dV + self.rho * self.V),
                              scale * (self.dWs + self.rho * self.Ws), scale * self.dbs]

    def forward_prop(self, tree, test=False):
        cost = self.forward_prop_node(tree.root)
        if not test:
            tree.mask = np.random.binomial(1, self.keep, self.mem_dim)
            theta = self.Ws.dot(tree.root.hActs1 * tree.mask) + self.bs
        else:
            theta = self.Ws.dot(tree.root.hActs1) * self.keep + self.bs
        theta -= np.max(theta)
        theta[theta < -500] = -500
        tree.probs = np.exp(theta)
        tree.probs /= np.sum(tree.probs)

        cost += -np.log(tree.probs[tree.label])
        return cost, np.argmax(tree.probs)

    def forward_prop_node(self, node, depth=-1):
        cost = 0.0
        if node.isLeaf:
            node.c = self.W_in.dot(self.L[:, node.word]) + self.b_in
            node.o = sigmoid(self.W_out.dot(self.L[:, node.word]) + self.b_out)
            node.ct = np.tanh(node.c)
            node.hActs1 = node.o * node.ct
        else:
            cost_left = self.forward_prop_node(node.left, depth - 1)
            cost_right = self.forward_prop_node(node.right, depth - 1)
            cost += (cost_left + cost_right)
            children = np.hstack((node.left.hActs1, node.right.hActs1))
            node.i = sigmoid(self.Ui.dot(children) + self.bi)
            node.f_l = sigmoid(self.Uf_l.dot(children) + self.bf)
            node.f_r = sigmoid(self.Uf_r.dot(children) + self.bf)
            node.o = sigmoid(self.Uo.dot(children) + self.bo)
            node.u = np.tanh(self.Uu.dot(children) + self.bu + children.dot(self.V).dot(children))
            node.c = node.i * node.u + node.f_l * node.left.c + node.f_r * node.right.c
            node.ct = np.tanh(node.c)
            node.hActs1 = node.o * node.ct
        return cost

    def back_prop(self, tree):
        deltas = tree.probs.copy()
        deltas[tree.label] -= 1.0
        self.dWs += np.outer(deltas, tree.root.hActs1 * tree.mask)
        self.dbs += deltas
        deltas = deltas.dot(self.Ws) * tree.mask
        self.back_prop_node(tree.root, deltas)
        pass

    def back_prop_node(self, node, errorH, errorC=None, depth=-1):
        errorO = errorH * node.ct * node.o * (1 - node.o)
        if errorC is None:
            errorC = errorH * node.o * (1 - node.ct ** 2)
        else:
            errorC += errorH * node.o * (1 - node.ct ** 2)
        if node.isLeaf:
            self.dW_out += np.outer(errorO, self.L[:, node.word])
            self.db_out += errorO
            self.dW_in += np.outer(errorC, self.L[:, node.word])
            self.db_in += errorC
            self.dL[node.word] += errorO.dot(self.W_out) + errorC.dot(self.W_in)
        else:
            children = np.hstack((node.left.hActs1, node.right.hActs1))
            self.dbo += errorO
            self.dUo += np.outer(errorO, children)
            errorDownH = errorO.dot(self.Uo)
            errorI = errorC * node.u * node.i * (1 - node.i)
            self.dbi += errorI
            self.dUi += np.outer(errorI, children)
            errorDownH += errorI.dot(self.Ui)
            errorU = errorC * node.i * (1 - node.u ** 2)
            self.dbu += errorU
            self.dUu += np.outer(errorU, children)
            self.dV += np.multiply.outer(errorU, np.outer(children, children))
            errorDownH += errorU.dot(self.Uu + (self.V + self.V.transpose((0, 2, 1))).dot(children))
            errorFL = errorC * node.left.c * node.f_l * (1 - node.f_l)
            errorFR = errorC * node.right.c * node.f_r * (1 - node.f_r)
            self.dbf += (errorFL + errorFR)
            self.dUf_l += np.outer(errorFL, children)
            self.dUf_r += np.outer(errorFR, children)
            errorDownH += (errorFL.dot(self.Uf_l) + errorFR.dot(self.Uf_r))
            errorCL = errorC * node.f_l
            errorCR = errorC * node.f_r

            self.back_prop_node(node.left, errorDownH[:self.mem_dim], errorCL, depth - 1)
            self.back_prop_node(node.right, errorDownH[self.mem_dim:], errorCR, depth - 1)

    def update_params(self, scale, update, log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P, dP in zip(self.stack[1:], update[1:]):
                pRMS = np.sqrt(np.mean(P ** 2))
                dpRMS = np.sqrt(np.mean((scale * dP) ** 2))
                print "weight rms=%f -- update rms=%f" % (pRMS, dpRMS)

        self.stack[1:] = [P + scale * dP for P, dP in zip(self.stack[1:], update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:, j] += scale * dL[j]

    def to_file(self, fid):
        pickle.dump(self.stack, fid)

    def from_file(self, fid):
        self.stack = pickle.load(fid)

    def check_grad(self, data, epsilon=1e-6):
        state =  np.random.get_state()
        cost, grad = self.cost_and_grad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dW..."
        for W, dW in zip(self.stack[1:], grad[1:]):
            W = W[..., None, None]  # add dimension since bias is flat
            dW = dW[..., None, None]
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i, j, k] += epsilon
                        np.random.set_state(state)
                        costP, _ = self.cost_and_grad(data)
                        W[i, j, k] -= epsilon
                        numGrad = (costP - cost) / epsilon
                        err = float(np.abs(dW[i, j, k] - numGrad))
                        print "Analytic %.9f, Numerical %.9f, Relative Error %.9f" % (dW[i, j, k], numGrad, err)
                        err1 += err
                        count += 1

        if 0.001 > err1:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1 / count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i, j] += epsilon
                np.random.set_state(state)
                costP, _ = self.cost_and_grad(data)
                L[i, j] -= epsilon
                numGrad = (costP - cost) / epsilon
                err = np.abs(dL[j][i] - numGrad)
                # print "Analytic %.9f, Numerical %.9f, Relative Error %.9f" % (dL[j][i], numGrad, err)
                err2 += err
                count += 1

        if 0.001 > err2:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2 / count)


if __name__ == '__main__':
    import loadTree as tree

    train = tree.load_trees('./data/train.json', tree.aspect_label)
    training_word_map = tree.load_word_map()
    numW = len(training_word_map)
    tree.convert_trees(train, training_word_map)

    wvecDim = 10
    outputDim = 5
    memDim = 5

    rnn = TreeTLSTM(wvecDim, memDim, outputDim, numW, mb_size=4)
    rnn.init_params()

    mbData = train[:5]

    print "Numerical gradient check..."
    rnn.check_grad(mbData, 1e-7)
