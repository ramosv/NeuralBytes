import math
from .utils import initialize_empty_matrix

def sigmoid(Z):
    """
    Sigmoid activation function but with a limit set at 15 and -15 for Z values.
    """
    A = initialize_empty_matrix(len(Z), len(Z[0]))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            val = Z[i][j]
            if val > 15:
                val = 15
            elif val < -15:
                val = -15
            A[i][j] = round(1.0 / (1.0 + math.exp(-val)), 4)
    return A

def sigmoid_derivative(A):
    """
    Given a matrix A containing sigmoid outputs,
    return its elementwise derivative.
    """
    deriv = initialize_empty_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            deriv[i][j] = A[i][j] * (1 - A[i][j])
    return deriv

def relu(Z):
    """
    ReLU activation function. 
    ReLU(x) = max(0, x).
    """
    A = initialize_empty_matrix(len(Z), len(Z[0]))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            A[i][j] = max(0, Z[i][j])
    return A

def relu_derivative(Z):
    """
    Given a matrix Z (the pre-activation inputs),
    return the derivative of ReLU(Z).
    Derivative of ReLU(x) = 1 if x > 0, else 0.
    
    Note: Some implementations pass the post-activation
    A into relu_derivative(A), but here we assume Z is 
    the same shape pre-activation matrix.
    """
    deriv = initialize_empty_matrix(len(Z), len(Z[0]))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            deriv[i][j] = 1 if Z[i][j] > 0 else 0
    return deriv

def softmax(Z):
    """
    Compute softmax for a 2D list Z (assumed shape is: num_classes x 1)
    """
    flat = []
    for i in range(len(Z)):
        flat.append(Z[i][0])
    
    max_val = flat[0]
    for val in flat:
        if val > max_val:
            max_val = val
            
    exp_vals = []
    for i in range(len(flat)):
        exp_val = math.exp(flat[i] - max_val)
        exp_vals.append(exp_val)
    
    sum_exp = 0
    for i in range(len(exp_vals)):
        sum_exp += exp_vals[i]
    
    softmax_values = []
    for i in range(len(exp_vals)):
        softmax_values.append(exp_vals[i] / sum_exp)
    
    output = initialize_empty_matrix(len(softmax_values), 1)
    for i in range(len(softmax_values)):
        output[i][0] = softmax_values[i]
    
    return output

def tanh(Z):
    """Hyperbolic tangent"""
    A = initialize_empty_matrix(len(Z), len(Z[0]))
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            A[i][j] = round(math.tanh(Z[i][j]), 4)
    return A


def tanh_derivative(A):
    """Element-wise derivative of tanh given post activation outputs A."""
    deriv = initialize_empty_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            deriv[i][j] = 1 - A[i][j] ** 2
    return deriv