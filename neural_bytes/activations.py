import math
from .utils import initialize_empty_matrix

def sigmoid(Z):
    """
    sigmoid activation function but with a limit set at 15 and -15 for Z values.
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

    deriv = initialize_empty_matrix(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            deriv[i][j] = A[i][j] * (1 - A[i][j])
    return deriv

# For future work, not being used right now!
# def relu(Z):
    
#     A = initialize_empty_matrix(len(Z), len(Z[0]))
#     for i in range(len(Z)):
#         for j in range(len(Z[0])):
#             A[i][j] = max(0, Z[i][j])

#     return A

# def relu_derivative(A):
    
#     deriv = initialize_empty_matrix(len(A), len(A[0]))

#     for i in range(len(A)):
#         for j in range(len(A[0])):
#             deriv[i][j] = 1 if A[i][j] > 0 else 0
#     return deriv

