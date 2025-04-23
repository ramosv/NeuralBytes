from neural_bytes.utils import rgb_to_grey
import pickle
from neural_bytes import CNN
import random

def unpickle(file):
    """Load a CIFAR-10 batch file using Python's pickle."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

if __name__ == "__main__":
    cifar_path = "C:/Users/ramos/Desktop/GitHub/NeuralBytes/cifar-10-batches-py"
    data_dict = unpickle(f"{cifar_path}/data_batch_1")
    all_images = data_dict[b'data']
    all_labels = data_dict[b'labels']

    num_total = 2000
    images = all_images[:num_total]
    labels = all_labels[:num_total]

    X = []
    for i in range(num_total):
        row = images[i]
        R = row[0:1024]
        G = row[1024:2048]
        B = row[2048:3072]

        rgb_image = []
        for rr in range(32):
            row_list = []
            for cc in range(32):
                idx = rr * 32 + cc
                row_list.append([R[idx], G[idx], B[idx]])
            rgb_image.append(row_list)


    Y = labels

    indices = list(range(num_total))
    random.shuffle(indices)
    split_idx = int(0.7 * num_total)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    trainX = []
    trainY = []
    for i in train_indices:
        trainX.append(X[i])
        trainY.append(Y[i])

    testX = []
    testY = []
    for i in test_indices:
        testX.append(X[i])
        testY.append(Y[i])


    cnn = CNN(trainX, trainY, filter_size=(3,3), num_filters=4, epochs=50, lr=1e-3, num_classes=10)
    cnn.train()

    correct = 0
    for i in range(len(testX)):
        pred_class = cnn.predict(testX[i])
        if pred_class == testY[i]:
            correct += 1
    accuracy = correct / len(testX)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    idx_test = 0
    print("Example test image:")
    print(f"Predicted = {pred_class}, Actual = {testY[idx_test]}")
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Predicted class = {classes[pred_class]}, Actual = {classes[testY[idx_test]]}")