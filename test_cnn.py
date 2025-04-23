from pathlib import Path
import pickle, random
from neural_bytes.utils import rgb_to_grey
from neural_bytes import CNN

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

if __name__ == "__main__":
    cifar_dir = Path("./cifar-10-batches-py")
    num_per_batch = 100

    images = []
    labels = []

    # fist 100 from each batch to get more distribution of classes
    for batch_idx in range(1, 6):
        data = unpickle(cifar_dir / f"data_batch_{batch_idx}")
        batch_imgs = data[b"data"][:num_per_batch]
        batch_lbls = data[b"labels"][:num_per_batch]

        for raw_vec in batch_imgs:
            R = raw_vec[0:1024]
            G = raw_vec[1024:2048]
            B = raw_vec[2048:3072]

            rgb = []
            for r in range(32):
                row = []
                for c in range(32):
                    idx = r*32 + c
                    row.append([R[idx], G[idx], B[idx]])
                rgb.append(row)

            images.append(rgb)

        labels.extend(batch_lbls)

    X = []
    for img in images:
        X.append(rgb_to_grey(img))

    Y = labels

    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(0.7 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    trainX = []
    trainY = []
    for i in train_idx:
        trainX.append(X[i])
        trainY.append(Y[i])

    testX = []
    testY = []
    for i in test_idx:
        testX.append(X[i])
        testY.append(Y[i])

    cnn = CNN(trainX, trainY, filter_size=(3,3), num_filters=4, epochs=50, lr=1e-3, num_classes=10)
    cnn.train()

    correct = 0
    for i in range(len(testX)):
        if cnn.predict(testX[i]) == testY[i]:
            correct += 1

    acc = correct / len(testX) * 100
    print(f"Test Accuracy: {acc:.2f}%")
