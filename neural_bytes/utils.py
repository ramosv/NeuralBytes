
def matrix_multiplication(A, B):

    if len(A[0]) == len(B):
        M = initialize_empty_matrix(len(A), len(B[0]))
        # We can multiply using regular row x col matrix multiplication
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(A[0])):
                    M[i][j] += A[i][k] * B[k][j]
    else:
        raise ValueError("Shapes of A and B do not match for matrix multiplication")

    return M

def initialize_empty_matrix(r, c):

    M = []
    row = [0] * c

    for i in range(r):
        M.append(list(row))

    return M

def transpose(A):

    B = initialize_empty_matrix(len(A[0]), len(A))
    for i in range(len(A)):
        for j in range(len(A[0])):
            B[j][i] = A[i][j]

    return B


def elementwise_add(A, B):

    C = initialize_empty_matrix(len(A), len(A[0]))
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        # we can then add the elements
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] + B[i][j]

    else:
        raise ValueError(f"Shape of A: {A} and B: {B} does ont match")
    return C


def elementwise_substraction(A, B):

    C = initialize_empty_matrix(len(A), len(A[0]))
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        # we can then sub the elements
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] - B[i][j]

    else:
        raise ValueError(f"Shape of A: {A} and B: {B} does ont match")
    return C


def elementwise_mul(A, B):

    M = initialize_empty_matrix(len(A), len(A[0]))
    if (len(A) == len(B)) and (len(A[0]) == len(B[0])):
        # we can use hadamard product of two matrices
        for i in range(len(A)):
            for j in range((A[0])):
                M[i][j] = A[i][j] * B[i][j]

    else:
        raise ValueError(f"Shape of A and B does not match")

    return M