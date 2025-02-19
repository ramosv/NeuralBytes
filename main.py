from neural_bytes import nn

if __name__ == "__main__":
    # test the functions
    X = [[0, 0, 1, 1], [0, 1, 0, 1]]
    Y = [0, 1, 1, 1]
    epochs = 10
    lr = 0.5

    #W1, b1, W2, b2 = nn.train(X, Y, epochs, lr)
    print(f"Final W1: {W1}\nFinal b1: {b1}\nFinal W2: {W2}\nFinal b2: {b2}")
