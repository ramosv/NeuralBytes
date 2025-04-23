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


def elementwise_multiplication(A, B):

    M = initialize_empty_matrix(len(A), len(A[0]))
    if (len(A) == len(B)) and (len(A[0]) == len(B[0])):
        # we can use hadamard product of two matrices
        for i in range(len(A)):
            for j in range(len(A[0])):
                M[i][j] = A[i][j] * B[i][j]

    else:
        raise ValueError(f"Shape of A and B does not match")

    return M

def scale_matrix(A, scalar):
        """
        Multiply every element of matrix A by 'scalar'.
        """
        M = initialize_empty_matrix(len(A), len(A[0]))
        for i in range(len(A)):
            for j in range(len(A[0])):
                M[i][j] = A[i][j] * scalar
        return M

def flatten(M):
    # flatten a 2D list into 2D list of shape n,1

    flattened = []
    for row in M:
        for val in row:
            flattened.append([val])

    return flattened

def rgb_to_grey(image):
    # conver rbg image to grey
    imageHeight = len(image)
    imageWidth = len(image[0])

    grey_img = initialize_empty_matrix(imageHeight, imageWidth)

    for i in range(imageHeight):
        for j in range(imageWidth):
            # this is the stanndard convertion values, I found online
            grey_img[i][j] = float(image[i][j][0] * 0.2126 +
                                 image[i][j][1] * 0.7152 + 
                                 image[i][j][2] * 0.0722)
            grey_img[i][j] /= 255.0
            
    return grey_img

def build_ascii_vocab(ascii_chars=256):
    """
    Build mappings for every ascci char 0-255.
    """
    char_to_ix = {}
    ix_to_char = {}
    for i in range(ascii_chars):
        ch = chr(i)
        char_to_ix[ch] = i
        ix_to_char[i] = ch
    return char_to_ix, ix_to_char

def text_to_indices(text, char_to_ix):
    """
    Convert a string into a list of integer codes or 0s.
    """
    indices = []
    for ch in text:
        if ch in char_to_ix:
            indices.append(char_to_ix[ch])
        else:
            indices.append(0)
    return indices

def build_word_vocab(text, max_words=5000):
    words = text.split()
    freq = {}
    for w in words:
        if w in freq:
            freq[w] += 1
        else:
            freq[w] = 1

    # sort the words by frequency in descending order
    def sort_by_freq(item):
        return item[1]

    items = list(freq.items())
    items.sort(key=sort_by_freq, reverse=True)

    vocab = []
    for i in range(min(max_words, len(items))):
        vocab.append(items[i])

    word_to_ix = {}
    ix_to_word = {}
    idx = 0
    for w, _ in vocab:
        word_to_ix[w] = idx
        ix_to_word[idx] = w
        idx += 1

    return word_to_ix, ix_to_word

def text_to_word_indices(text, word_to_ix):
    seq = []
    for w in text.split():
        if w in word_to_ix:
            seq.append(word_to_ix[w])
        else:
            seq.append(0)

    return seq