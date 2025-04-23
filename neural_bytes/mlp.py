import random
import math
from .utils import initialize_empty_matrix, matrix_multiplication, transpose, elementwise_substraction, elementwise_multiplication
from .activations import sigmoid, sigmoid_derivative

class MLP():
    def __init__(self, X, Y, epochs=100, lr=1e-1, hidden_layer=2):
        """
        Initialize parameters.
        For now the NN only supports up to 2-neuron hidden layers.
        X: (2, number of samples)
        Y: (1, number of samples)
        """

        self.X = X
        self.Y =Y
        self.lr = lr
        self.epochs = epochs
        self.hidden_layers = hidden_layer

    def _initialize_parameters(self):
        """
        Returns a list of tuples with (weights, bias)
        """
        weights_bias = []
        
        for _ in range(self.hidden_layers):
            w = [[(random.random() - 0.5)*0.2, (random.random() - 0.5)*0.2],[(random.random() - 0.5)*0.2, (random.random() - 0.5)*0.2]]
            b = [[(random.random() - 0.5)*0.2],[(random.random() - 0.5)*0.2]]
            weights_bias.append((w, b))
        
        last_weight = [[(random.random() - 0.5)*0.2, (random.random() - 0.5)*0.2]]
        last_bias = [[(random.random() - 0.5)*0.2]]
        weights_bias.append((last_weight, last_bias))

        return weights_bias

        
    def _bias_addition(self, Z, b):
        """
        Adds bias b to each column of matrix Z.
        Z: (rows, columns)
        b: (rows, 1)
        """

        rows = len(Z)
        cols = len(Z[0])

        M = initialize_empty_matrix(rows, cols)
        if len(b) != rows:
            raise ValueError(f"number of rows in Z: {rows} != number of rows in b: {len(b)}")

        for i in range(rows):
            for j in range(cols):
                M[i][j] = Z[i][j] + b[i][0]

        return M
        
    def _forward_pass(self, weights_bias):
        """
        Forward pass through the network.
        For each layer we will do the following
        - Compute Z = matrix_multiplication(W, A) then add bias b.
        - Apply the sigmoid activation to get A.
        - Save the tuple (Z, A) in z_and_a.
        """

        A = self.X
        z_and_a = []
        for i, (w,b) in enumerate(weights_bias):
            w,b = weights_bias[i]
            Z = matrix_multiplication(w, A)
            Z = self._bias_addition(Z, b)
            A = sigmoid(Z)
            z_and_a.append((Z,A))
        
        return z_and_a
    
    def _compute_cost(self, A_out):
        """
        Compute MSE cost.
        A_out: (1, n_samples)
        Y: (1, n_samples)
        """

        sum_sq = 0
        
        for i in range(len(self.Y[0])):
            sum_sq += (A_out[0][i] - self.Y[0][i]) ** 2

        cost = sum_sq /(2 * len(self.Y[0]))
        return cost
    

    def _sum_layer_error(self, A):
        """
        Sums the elements of A over the samples 
        """
        n = len(A)
        m = len(A[0])
        result = []
        for i in range(n):
            s = 0
            for j in range(m):
                s += A[i][j]
            result.append([s])
        return result


    def _multiply_lr(self, A, scalar):
        """
        Multiplies each element of matrix A by a scalar.
        """

        M = initialize_empty_matrix(len(A), len(A[0]))

        for i in range(len(A)):
            for j in range(len(A[0])):
                M[i][j] = A[i][j] * scalar

        return M


    def _backward_pass(self, z_and_a, weights_bias):
        """
        Performs backpropagation to compute gradients for each layer.
        The output layer uses elementwise_multiplication(a - Y, sigmoid_derivative(z))
        For hidden layers the gradient is computed using the transposed weight matrix.
        and the delta from the next layer.
        """
        layer_errors = []
        m = len(self.Y[0])
        
        # output layer
        Z_out, A_out = z_and_a[-1]

        # using the activation from the previous layer or the input X 
        A_prev = None

        if len(z_and_a) > 1:
            A_prev = z_and_a[-2][1]
        else:
            A_prev = self.X

        # delta and gradients for the output layer
        delta = elementwise_multiplication(elementwise_substraction(A_out, self.Y), sigmoid_derivative(Z_out))
        dW = self._multiply_lr(matrix_multiplication(delta, transpose(A_prev)), 1/m)
        db = self._multiply_lr(self._sum_layer_error(delta), 1/m)
        layer_errors.append((dW, db))
        prev_delta = delta

        # hidden layers
        for layer in range(len(z_and_a)-2, -1, -1):
            Z_curr, A_curr = z_and_a[layer]

            # for the first layer
            if layer == 0:
                A_prev = self.X
            else:
                A_prev = z_and_a[layer - 1][1]
            
            # delta and gradients
            delta = elementwise_multiplication(matrix_multiplication(transpose(weights_bias[layer+1][0]), prev_delta),sigmoid_derivative(Z_curr))
            dW = self._multiply_lr(matrix_multiplication(delta, transpose(A_prev)), 1/m)
            db = self._multiply_lr(self._sum_layer_error(delta), 1/m)
            layer_errors.insert(0, (dW, db))
            prev_delta = delta

        return layer_errors

    def _update_parameters(self, weights_bias, layer_error):
        new_weights_bias = []

        # layer_error and weights_bias are in reverse order already
        for i in range(len(weights_bias)):
            w,b = weights_bias[i]
            dW, db = layer_error[i]
            new_weights = elementwise_substraction(w, self._multiply_lr(dW, self.lr))
            new_bias = elementwise_substraction(b, self._multiply_lr(db, self.lr))
            new_weights_bias.append((new_weights, new_bias))

        return new_weights_bias
    
    def predict(self, X, weights_bias):
        """
        Predict output for input x using weights and biases
        """
        A = X
        for (w, b) in weights_bias:
            Z = matrix_multiplication(w, A)
            Z = self._bias_addition(Z, b)
            A = sigmoid(Z)
        return A

    def train(self):
        """
        Trains the neural network
        For each epoch we do the following.
        - Perform a forward pass.
        - Compute the cost using the output activation.
        - Do backpropagation to compute gradients.
        - Update the network parameters.
        """
        weights_bias = self._initialize_parameters()
        cost_history = []

        for e in range(self.epochs):
            z_and_a = self._forward_pass(weights_bias)
            cost = self._compute_cost(z_and_a[-1][1])

            # in case we need an early stopping
            if math.isnan(cost):
                print(f"Cost is NaN at epoch: {e}")
                break

            cost_history.append(cost)
            layer_errors = self._backward_pass(z_and_a, weights_bias)
            weights_bias = self._update_parameters(weights_bias, layer_errors)
            if (e + 1) % 100 == 0:
                print("Epoch:", e + 1, "Cost:", cost)

        return weights_bias, cost_history
    