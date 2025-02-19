import random
import math
from .utils import initialize_empty_matrix, matrix_multiplication, transpose, elementwise_substraction
from .activations import sigmoid, sigmoid_derivative

class NN():
    def __init__(self, X, Y, epochs=10, lr=0.0001, hidden_layer=2):
        """
        Initialize parameters
        """

        self.X = X
        self.Y =Y
        self.lr = lr
        self.epochs = epochs
        self.hidden_layers = hidden_layer

    def _initialize_parameters(self):
        """
        The current architecture support from 2 to n hidden layers
        returns a list of tuples with (weights, bias) 
        """
        weights_bias = []
        for _ in range(len(self.hidden_layers -1)):
            w = [[random.random(), random.random()], [random.random(), random.random()]]
            b = [[random.random()], [random.random()]]
            weights_bias.append((w, b))
        
        # last layer
        last_weight = [[random.random()], [random.random()]]
        last_bias = random.random()
        weights_bias.append((last_weight, last_bias))
        return weights_bias
        
    def _bias_addition(self, A, W):

        M = initialize_empty_matrix(len(A), len(A[0]))
        if len(W) != len(A):
            raise ValueError(f"number of rows in A: {A} and W: {W} does not match")
        else:
            for i in range(len(A)):
                for j in range(len(A[0])):
                    M[i][j] = A[i][j] + W[i][0]

            return M

    def _sum_layer_error(self, A):
        """
        Sum elements of A along an axis.
        return a matrix of shape (1, len(A[0]))
        """
        size = len(A[0])
        col_sums = [0] * size

        for row in A:
            for i in range(size):
                col_sums[i] += row[i]

        # a 1 x size matrix
        return [col_sums]


    def _multiply_lr(self, A):

        M = initialize_empty_matrix(len(A), len(A[0]))

        for i in range(len(A)):
            for j in range(len(A[0])):
                M[i][j] = A[i][j] * self.lr

        return M

    def _forward_pass(self, weights_bias):
        """
        Forward pass throught the network
        """
        z_and_a = []
        for i in range(len(weights_bias)):
            w,b = weights_bias[i]
            z = matrix_multiplication(w, self.X[i])
            z = self._bias_addition(z,b)
            a = sigmoid(z)
            z_and_a.append(z,a)
        
        return z_and_a


    def _compute_cost(self, z_and_a):
        # To compute the cost of the output layer
        # A2 is the output of the second layer
        # Y is the label for each sample
        # n is the number of samples

        # output layer is still in matrix form, therefore we need to flatten or take first row

        diff_sq_sum = 0
        for a, y in zip(z_and_a[-1], self.Y):
            diff_sq_sum += (a - y) ** 2
        cost = (1 / (2 * len(self.Y))) * diff_sq_sum
        return cost


    def _backward_pass(self, z_and_a, weights_bias):
        # output layer error
        # A2 is the output of the second layer which is a matrix so we need to take the first row
        # given a neural network with a depth of 2 neurons. The dimmension of the output layer is always to be 1
    
        # as part of pythons standard library we can use the reversed funcin to iterate from the end

        layer_errors = []
        prev_delta = None


        for layer in reversed(z_and_a):
            # the weights and bias from the layer starting from the end
            xy_index = z_and_a.index(layer)
            z, a = layer

            # the last hidden layer is always the output layer
            if layer == z_and_a[-1]:
                delta = (a[0] - self.Y) * sigmoid_derivative(z)
                prev_delta = delta

            # middle hidden layers
            elif xy_index != 0:
                delta = matrix_multiplication(transpose(weights_bias[xy_index]),prev_delta) * sigmoid_derivative(z)

                delta_weights = (1 / len(self.Y)) * matrix_multiplication(a,transpose(prev_delta))
                delta_bias = (1 / len(self.Y) * self._sum_layer_error(prev_delta))
                prev_delta = delta
                layer_errors.append((delta_weights, delta_bias))
            # input layer
            else:
                delta_weights = (1 / len(self.Y)) * matrix_multiplication(a,transpose(prev_delta))
                delta_bias = (1 / len(self.Y) * self._sum_layer_error(prev_delta))
                layer_errors.append((delta_weights, delta_bias))

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

    def train(self):
        # Initialize parameters
        weighs_bias = self._initialize_parameters()

        for e in range(self.epochs):
            # forward pass
            z_and_a = self._forward_pass(weighs_bias)

            # cost
            cost = self._compute_cost(z_and_a)

            # backward prop
            layer_errors = self._backward_pass(z_and_a, weighs_bias)

            # 5) update
            weighs_bias = self._update_parameters(weighs_bias, layer_errors)

            # print cost
            if (e + 1) % 100 == 0:
                print("Epoch:", e + 1, "Cost:", cost)

        # Return final parameters
        return weighs_bias


# def hyperparameter_tuning():
#     config = {
#         "epochs": [100,200,300,400,500],
#         "lr": [1e-0.4, 1e-0.2],
#         "hidden_layer": [2, 3, 4],
#     }