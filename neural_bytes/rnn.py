import math
import random
from .utils import initialize_empty_matrix, matrix_multiplication, transpose, elementwise_add, elementwise_multiplication
from .activations import tanh, tanh_derivative, softmax

class RNN:
    """
    Char lvl RNN with one hidden layer and BPTT.
    """
    def __init__(self, vocab_size, hidden_size=128, seq_length=25, learning_rate=0.1, clip_value=5.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        # params are U, W, V, b, c
        self.U = initialize_empty_matrix(hidden_size, vocab_size)
        self.W = initialize_empty_matrix(hidden_size, hidden_size)
        self.V = initialize_empty_matrix(vocab_size, hidden_size)
        self.b = initialize_empty_matrix(hidden_size, 1)
        self.c = initialize_empty_matrix(vocab_size, 1)

        # memoryory
        self.mU = initialize_empty_matrix(hidden_size, vocab_size)
        self.mW = initialize_empty_matrix(hidden_size, hidden_size)
        self.mV = initialize_empty_matrix(vocab_size, hidden_size)
        self.mb = initialize_empty_matrix(hidden_size, 1)
        self.mc = initialize_empty_matrix(vocab_size, 1)

        scale_U = math.sqrt(1.0 / vocab_size)
        scale_W = math.sqrt(1.0 / hidden_size)

        for i in range(hidden_size):
            for j in range(vocab_size):
                self.U[i][j] = (random.random() - 0.5) * 2 * scale_U

        for i in range(hidden_size):
            for j in range(hidden_size):
                self.W[i][j] = (random.random() - 0.5) * 2 * scale_W

        for i in range(vocab_size):
            for j in range(hidden_size):
                self.V[i][j] = (random.random() - 0.5) * 2 * scale_W

    def forward(self, inputs, targets, hprev):
        # One-hot encoding and forward pass
        xs, hs, ps = {}, {}, {}
        hs[-1] = []
        for row in hprev:
            hs[-1].append(row[:])
        loss = 0.0

        for t in range(len(inputs)):
            # one hot
            x_t = initialize_empty_matrix(self.vocab_size, 1)
            code = inputs[t]
            if 0 <= code < self.vocab_size:
                x_t[code][0] = 1.0
            xs[t] = x_t

            # hidden state
            z1 = elementwise_add(matrix_multiplication(self.U, x_t), matrix_multiplication(self.W, hs[t-1]))
            z1b = elementwise_add(z1, self.b)
            h_t = tanh(z1b)
            hs[t] = h_t

            # output
            z2 = elementwise_add(matrix_multiplication(self.V, h_t), self.c)
            p_t = softmax(z2)
            ps[t] = p_t

            # loss
            target = targets[t]
            if 0 <= target < self.vocab_size:
                loss -= math.log(p_t[target][0] + 1e-12)

        cache = {"xs": xs, "hs": hs, "ps": ps, "targets": targets}
        return loss, cache, hs[len(inputs)-1]

    def backward(self, cache):
        xs, hs, ps, targets = cache["xs"], cache["hs"], cache["ps"], cache["targets"]

        # initialize gradients, just empty matrices for now
        dU = initialize_empty_matrix(self.hidden_size, self.vocab_size)
        dW = initialize_empty_matrix(self.hidden_size, self.hidden_size)
        dV = initialize_empty_matrix(self.vocab_size, self.hidden_size)
        db = initialize_empty_matrix(self.hidden_size, 1)
        dc = initialize_empty_matrix(self.vocab_size, 1)
        dhnext = initialize_empty_matrix(self.hidden_size, 1)

        # bptt
        for t in reversed(range(len(xs))):
            # softmax
            dy = []
            for row in ps[t]:
                dy.append(row[:])

            idx = targets[t]
            if 0 <= idx < self.vocab_size:
                dy[idx][0] -= 1.0

            # output layer
            dV = elementwise_add(dV, matrix_multiplication(dy, transpose(hs[t])))
            dc = elementwise_add(dc, dy)

            # put it back into hidden
            dh = elementwise_add(matrix_multiplication(transpose(self.V), dy), dhnext)

            # tanh backprop
            dh_raw = elementwise_multiplication(tanh_derivative(hs[t]), dh)
            db = elementwise_add(db, dh_raw)
            dU = elementwise_add(dU, matrix_multiplication(dh_raw, transpose(xs[t])))
            dW = elementwise_add(dW, matrix_multiplication(dh_raw, transpose(hs[t-1])))

            # carry to next
            dhnext = matrix_multiplication(transpose(self.W), dh_raw)

        # clip it
        for grad in (dU, dW, dV, db, dc):
            for i in range(len(grad)):
                for j in range(len(grad[0])):
                    if grad[i][j] > self.clip_value: grad[i][j] = self.clip_value
                    if grad[i][j] < -self.clip_value: grad[i][j] = -self.clip_value

        return dU, dW, dV, db, dc

    def update(self, grads):
        dU, dW, dV, db, dc = grads

        for param, dparam, memory in zip(
            (self.U, self.W, self.V, self.b, self.c),
            (dU, dW, dV, db, dc),
            (self.mU, self.mW, self.mV, self.mb, self.mc)):
            
            for i in range(len(memory)):
                for j in range(len(memory[0])):
                    memory[i][j] += dparam[i][j] * dparam[i][j]
                    param[i][j] -= self.learning_rate * dparam[i][j] / math.sqrt(memory[i][j] + 1e-8)

    def sample(self, seed_ix, n, h=None):
        if h is None:
            h = initialize_empty_matrix(self.hidden_size, 1)

        x = initialize_empty_matrix(self.vocab_size, 1)
        if 0 <= seed_ix < self.vocab_size:
            x[seed_ix][0] = 1.0

        output = []
        for _ in range(n):
            z1 = elementwise_add(matrix_multiplication(self.U, x), matrix_multiplication(self.W, h))
            z1b = elementwise_add(z1, self.b)
            h = tanh(z1b)

            z2 = elementwise_add(matrix_multiplication(self.V, h), self.c)
            p = softmax(z2)

            r = random.random()
            cumulative = 0.0
            idx = 0

            for i in range(self.vocab_size):
                cumulative += p[i][0]
                if r < cumulative:
                    idx = i
                    break

            x = initialize_empty_matrix(self.vocab_size, 1)
            x[idx][0] = 1.0
            output.append(idx)

        return output
