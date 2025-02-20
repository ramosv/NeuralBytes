import math
from .utils import initialize_empty_matrix

def sigmoid(Z):

    A = initialize_empty_matrix(len(Z), len(Z[0]))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            A[i][j] = 1 / (1 + math.exp(-Z[i][j]))

    return A

def sigmoid_derivative(A):

    deriv = initialize_empty_matrix(len(A), len(A[0]))

    for i in range(len(A)):
        for j in range(len(A[0])):
            deriv[i][j] = A[i][j] * (1 - A[i][j])
    return deriv
