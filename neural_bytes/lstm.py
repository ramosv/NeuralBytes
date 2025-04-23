import math
import random
from .utils import (initialize_empty_matrix, matrix_multiplication, transpose, elementwise_add, elementwise_multiplication)
from .activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative, softmax

class LSTM:
    """
    char lvl LSTM with one hidden layer and BPTT.
    """
    def __init__(self, vocab_size=256, hidden_size=128, seq_length=25, learning_rate=0.1, clip_value=5.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        # forget gate
        self.Wf = initialize_empty_matrix(hidden_size, hidden_size)
        self.Uf = initialize_empty_matrix(hidden_size, vocab_size)
        self.bf = initialize_empty_matrix(hidden_size, 1)

        # input gate
        self.Wi = initialize_empty_matrix(hidden_size, hidden_size)
        self.Ui = initialize_empty_matrix(hidden_size, vocab_size)
        self.bi = initialize_empty_matrix(hidden_size, 1)

        # output gate
        self.Wo = initialize_empty_matrix(hidden_size, hidden_size)
        self.Uo = initialize_empty_matrix(hidden_size, vocab_size)
        self.bo = initialize_empty_matrix(hidden_size, 1)

        # cell candidate
        self.Wc = initialize_empty_matrix(hidden_size, hidden_size)
        self.Uc = initialize_empty_matrix(hidden_size, vocab_size)
        self.bc = initialize_empty_matrix(hidden_size, 1)

        # final output layer
        self.V = initialize_empty_matrix(vocab_size, hidden_size)
        self.c = initialize_empty_matrix(vocab_size, 1)

        # memory
        params = [self.Wf, self.Uf, self.bf,
                  self.Wi, self.Ui, self.bi,
                  self.Wo, self.Uo, self.bo,
                  self.Wc, self.Uc, self.bc,
                  self.V, self.c]
        
        self.memory = []
        for p in params:
            r, c = len(p), len(p[0])
            self.memory.append(initialize_empty_matrix(r, c))

        # random initialization
        for mat in [self.Wf, self.Wi, self.Wo, self.Wc]:
            for i in range(hidden_size):
                for j in range(hidden_size):
                    mat[i][j] = (random.random() - 0.5) * 0.1

        for mat in [self.Uf, self.Ui, self.Uo, self.Uc]:
            for i in range(hidden_size):
                for j in range(vocab_size):
                    mat[i][j] = (random.random() - 0.5) * 0.1

        for b in [self.bf, self.bi, self.bo, self.bc]:
            for i in range(hidden_size):
                b[i][0] = 0.0

        for i in range(vocab_size):
            for j in range(hidden_size):
                self.V[i][j] = (random.random() - 0.5) * 0.1

        for i in range(vocab_size):
            self.c[i][0] = 0.0

    def forward(self, inputs, targets, hprev, cprev=None):
        # initialize cprev if first call
        if cprev is None:
            cprev = initialize_empty_matrix(self.hidden_size, 1)

        xs, hs, cs = {}, {}, {}
        fs, ig, os, gs = {}, {}, {}, {}
        ys, ps = {}, {}

        hs[-1] = []
        for row in hprev:
            hs[-1].append(row[:])

        cs[-1] = []
        for row in cprev:
            cs[-1].append(row[:])
        loss = 0.0

        # time steps
        for t in range(len(inputs)):
            # one-hot x_t
            x_t = initialize_empty_matrix(self.vocab_size, 1)
            x_t[inputs[t]][0] = 1.0
            xs[t] = x_t
            # gates

            # sigmoid(Wf*h_{t-1} + Uf*x_t + bf)
            zf = elementwise_add(matrix_multiplication(self.Wf, hs[t-1]), matrix_multiplication(self.Uf, x_t))
            f_t = sigmoid(elementwise_add(zf,self.bf))
            
            # sigmoid(Wi*h_{t-1} + Ui*x_t + bi)
            zi = elementwise_add(matrix_multiplication(self.Wi, hs[t-1]), matrix_multiplication(self.Ui, x_t))
            i_t = sigmoid(elementwise_add( zi, self.bi ))

            # sigmoid(Wo*h_{t-1} + Uo*x_t + bo)
            zo = elementwise_add(matrix_multiplication(self.Wo, hs[t-1]), matrix_multiplication(self.Uo, x_t))
            o_t = sigmoid(elementwise_add(zo,self.bo))

            # cell candidate
            # tanh(Wc*h_{t-1} + Uc*x_t + bc)
            zg = elementwise_add(matrix_multiplication(self.Wc, hs[t-1]), matrix_multiplication(self.Uc, x_t))
            g_t = tanh(elementwise_add(zg,self.bc))

            # cell state
            c_t = elementwise_add(elementwise_multiplication(f_t, cs[t-1]),elementwise_multiplication(i_t, g_t))

            # hidden state
            tanh_c = tanh(c_t)
            h_t = elementwise_multiplication(o_t, tanh_c)

            # output
            y_t = elementwise_add(matrix_multiplication(self.V, h_t), self.c)
            p_t = softmax(y_t)
            loss -= math.log(p_t[targets[t]][0] + 1e-12)

            # save it
            fs[t], ig[t], os[t], gs[t] = f_t, i_t, o_t, g_t
            hs[t], cs[t], ys[t], ps[t] = h_t, c_t, y_t, p_t

        cache = {"xs":xs, "hs":hs, "cs":cs,
                 "fs":fs, "ig":ig, "os":os, "gs":gs,
                 "ys":ys, "ps":ps, "targets":targets}
        
        return loss, cache, hs[len(inputs)-1], cs[len(inputs)-1]

    def backward(self, cache):
        xs, hs, cs = cache["xs"], cache["hs"], cache["cs"]
        fs, ig, os, gs = cache["fs"], cache["ig"], cache["os"], cache["gs"]
        ps, targets = cache["ps"], cache["targets"]

        # init grads
        dWf = initialize_empty_matrix(self.hidden_size, self.hidden_size)
        dUf = initialize_empty_matrix(self.hidden_size, self.vocab_size)
        dbf = initialize_empty_matrix(self.hidden_size, 1)
        dWi = initialize_empty_matrix(self.hidden_size, self.hidden_size)
        dUi = initialize_empty_matrix(self.hidden_size, self.vocab_size)
        dbi = initialize_empty_matrix(self.hidden_size, 1)
        dWo = initialize_empty_matrix(self.hidden_size, self.hidden_size)
        dUo = initialize_empty_matrix(self.hidden_size, self.vocab_size)
        dbo = initialize_empty_matrix(self.hidden_size, 1)
        dWc = initialize_empty_matrix(self.hidden_size, self.hidden_size)
        dUc = initialize_empty_matrix(self.hidden_size, self.vocab_size)
        dbc = initialize_empty_matrix(self.hidden_size, 1)
        dV = initialize_empty_matrix(self.vocab_size, self.hidden_size)
        dc = initialize_empty_matrix(self.vocab_size, 1)
        dhnext = initialize_empty_matrix(self.hidden_size, 1)
        dcnext = initialize_empty_matrix(self.hidden_size, 1)

        # BPTT through time
        for t in reversed(range(len(xs))):
            # output softmax gradient
            dy = []
            for row in ps[t]:
                dy.append(row[:])

            dy[targets[t]][0] -= 1.0

            dV = elementwise_add(dV, matrix_multiplication(dy, transpose(hs[t])))
            dc = elementwise_add(dc, dy)

            # back into h and c
            dh = elementwise_add(matrix_multiplication(transpose(self.V), dy), dhnext)
            do = elementwise_multiplication(dh, tanh(cs[t]))
            do = elementwise_multiplication(do, sigmoid_derivative(os[t]))

            dWo = elementwise_add(dWo, matrix_multiplication(do, transpose(hs[t-1])))
            dUo = elementwise_add(dUo, matrix_multiplication(do, transpose(xs[t])))
            dbo = elementwise_add(dbo, do)

            # cell state gradient
            cell = elementwise_multiplication(dh, elementwise_multiplication(os[t], tanh_derivative(tanh(cs[t]))))
            dct = elementwise_add(cell, dcnext)

            # gates
            zf = elementwise_add( matrix_multiplication(self.Wf, hs[t-1]),matrix_multiplication(self.Uf, xs[t]))
            df = elementwise_multiplication(dct, cs[t-1])
            df = elementwise_multiplication(df, sigmoid_derivative( elementwise_add(zf, self.bf) ))

            dWf = elementwise_add(dWf, matrix_multiplication(df, transpose(hs[t-1])))
            dUf = elementwise_add(dUf, matrix_multiplication(df, transpose(xs[t])))
            dbf = elementwise_add(dbf, df)

            di = elementwise_multiplication(dct, gs[t])
            di = elementwise_multiplication(di, sigmoid_derivative(ig[t]))
            dWi = elementwise_add(dWi, matrix_multiplication(di, transpose(hs[t-1])))
            dUi = elementwise_add(dUi, matrix_multiplication(di, transpose(xs[t])))
            dbi = elementwise_add(dbi, di)

            dg = elementwise_multiplication(dct, ig[t])
            dg = elementwise_multiplication(dg, tanh_derivative(gs[t]))
            dWc = elementwise_add(dWc, matrix_multiplication(dg, transpose(hs[t-1])))
            dUc = elementwise_add(dUc, matrix_multiplication(dg, transpose(xs[t])))
            dbc = elementwise_add(dbc, dg)

            # propagate to previous
            temp = elementwise_add( matrix_multiplication(transpose(self.Wf), df), matrix_multiplication(transpose(self.Wi), di))
            temp1 = elementwise_add(matrix_multiplication(transpose(self.Wo), do),matrix_multiplication(transpose(self.Wc), dg))
            
            dhprev = elementwise_add(temp,temp1)
            dhnext = dhprev

            # cprev gradient
            dcnext = elementwise_multiplication(dct, fs[t])

        # clip all grads
        for grad in [dWf, dUf, dbf, dWi, dUi, dbi, dWo, dUo, dbo, dWc, dUc, dbc, dV, dc]:
            for i in range(len(grad)):
                for j in range(len(grad[0])):
                    if grad[i][j] > self.clip_value: grad[i][j] = self.clip_value
                    if grad[i][j] < -self.clip_value: grad[i][j] = -self.clip_value
        
        return (dWf, dUf, dbf,
                dWi, dUi, dbi,
                dWo, dUo, dbo,
                dWc, dUc, dbc,
                dV, dc)

    def update(self, grads):
        # adagrad update over all 14 params
        flat_params = [self.Wf, self.Uf, self.bf,
                       self.Wi, self.Ui, self.bi,
                       self.Wo, self.Uo, self.bo,
                       self.Wc, self.Uc, self.bc,
                       self.V,  self.c]
        
        for idx, param in enumerate(flat_params):
            grad = grads[idx]
            memory = self.memory[idx]

            for i in range(len(param)):
                for j in range(len(param[0])):
                    memory[i][j] += grad[i][j] * grad[i][j]
                    param[i][j] -= self.learning_rate * grad[i][j] / math.sqrt(memory[i][j] + 1e-8)
    
    def sample(self, seed_ix, n, h=None, c=None):
        if h is None:
            h = initialize_empty_matrix(self.hidden_size, 1)
        if c is None:
            c = initialize_empty_matrix(self.hidden_size, 1)

        x = initialize_empty_matrix(self.vocab_size, 1)
        x[seed_ix][0] = 1.0

        indices = []
        for _ in range(n):
            # forward one step
            zf = elementwise_add(matrix_multiplication(self.Wf, h), matrix_multiplication(self.Uf, x))
            f_t = sigmoid(elementwise_add(zf, self.bf))

            zi = elementwise_add(matrix_multiplication(self.Wi, h), matrix_multiplication(self.Ui, x))
            i_t = sigmoid(elementwise_add(zi, self.bi))

            zo = elementwise_add(matrix_multiplication(self.Wo, h), matrix_multiplication(self.Uo, x))
            o_t = sigmoid(elementwise_add(zo, self.bo))

            zg = elementwise_add(matrix_multiplication(self.Wc, h),matrix_multiplication(self.Uc, x))
            g_t = tanh(elementwise_add(zg, self.bc))

            # update cell state and hidden state
            c = elementwise_add(elementwise_multiplication(f_t, c), elementwise_multiplication(i_t, g_t))
            h = elementwise_multiplication(o_t, tanh(c))

            # compute output with softmax
            y = elementwise_add(matrix_multiplication(self.V, h), self.c)
            p = softmax(y)

            # sample from p
            r = random.random()
            cum = 0.0
            ix = 0
            for i in range(self.vocab_size):
                cum += p[i][0]
                if r < cum:
                    ix = i
                    break

            # prepare next input
            x = initialize_empty_matrix(self.vocab_size, 1)
            x[ix][0] = 1.0
            indices.append(ix)

        return indices
