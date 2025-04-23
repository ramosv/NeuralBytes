import math
import random
from neural_bytes.cnn import CNN
from neural_bytes.mlp import MLP


if __name__ == "__main__":
    X = [[0, 0, 1, 1], [0, 1, 0, 1]]
    Y = [[0, 1, 1, 1]]
    epochs = 100
    lr = 0.000001

    nn = MLP(X, Y, epochs=epochs, lr=lr, hidden_layer=1)
    final_params, cost_history = nn.train()

    new_predictions = []
    new_predictions.append(f"Epochs: {epochs} Learning Rate: {lr}\n")
    new_predictions.append("Predictions:\n")
    accuracy = 0
    for i in range(len(nn.X[0])):
        pt_input = [[nn.X[0][i]], [nn.X[1][i]]]
        pred = nn.predict(pt_input, final_params)[0][0]

        #in case is NaN
        if math.isnan(pred):
            pred = 0

        new_predictions.append(f"True = {nn.Y[0][i]} : Predicted = {pred:.4f}\n")
        #print(f"True label={nn.Y[0][i]} : Predicted by NN={pred:.5f}")

        accuracy += 1 
        if nn.Y[0][i] == round(pred):
            accuracy+=1
        else:
            accuracy+=0

    new_predictions.append(f"Accuracy: {accuracy/len(nn.X[0])}\n\n")

    #To avoid overwriting the existing stuff
    with open("predictions.txt", "r") as f:
        existing_content = f.read()

    with open("predictions.txt", "w") as f:
        f.writelines(new_predictions)
        f.write(existing_content)

    #visualize_nn(cost_history, nn, final_params, X, Y)




