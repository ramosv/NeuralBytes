# NeuralBytes

## Building a Neural Network from Scratch usuing only python native libraries.

Since I did not use numpy library. In order to perform matrix related operations, I build custom functions housed inside the utils.py dir.
Given the small data size, the neural network performs quite well, even with epochs as large as 100,000.

Other details on the network:
- The neural network (currently) only supports up to 2 neurons, but a N number of hidden layers.
- The NN class takes X, Y, epochs, and lr as initialization parameters.
- Python standard libraries used:
    - math
    - random
    - matplotlib: for visualization and to meet assignment requirements.

For testing and validating the network performance, I ran the following tests:

- Tested epcohs = [100,200,500,1000,2000,4000,6000,8000,10000].
- for each epoch, I tested the following learning rate = [0.1,0.01,0.001,0.0001,1e-05,1e-06].
- All output plots are availble in the plots directory.
- predictionss.txt offers a summary of my tests and additional information

### Short Answers

1. In my code, the gradient for each neuron is computed by following the chain rule over the sigmoid activation. I calculate the error at the output by finding the difference between the predicted values and the true labels. Then, I multiply that elementwise with the derivative of the sigmoid function to get the delta. This delta is propagated backward through matrix multiplications, where I also account for the inputs to each neuron, to obtain the gradients for weights and biases. All the functions related to matrix operations were built in house and are availble inside the utils file.

2. The sigmoid activation in my network introduces nonlinearity so that the model isn't just a stacking of linear operations. This nonlinearity enables the network to learn complex patterns and relationships in the data. Without it, regardless of the number of hidden layers, the network would behave like a single linear model.

3. Overfitting in my project would be evident if the network starts memorizing the training examples rather than generalizing. While the current implementation is simple (and the architecture is limited to 2-neuron hidden layers), I could counteract overfitting by using techniques like adding regularization terms to the cost function, implementing dropout in the hidden layers, or simply by gathering more training data.

4. If the network isn't able to capture the patterns in the data enough (i.e., underfitting), it might be because the network's capacity is too low. In this case, increasing the number of hidden layers or neurons per layer could help the network learn a more complex function, improving its performance.

### Hyperparater tunning
I experimented with different learning rates and hidden layer sizes. By adjusting these hyperparameters, I observed how they impacted the convergence and performance of the model. Plots dir an predictions.txt contains all my examples ran.

### Visualization
I plotted the decision boundaries after training the model to visualize how it separates the data. This helps in understanding the how the Neural Network perfoms with different learning rates and epochs.

## Questions and Challenges faced.

1. One of the main challenges for me was ensuring the correct calculation of gradients, keeping track of the previous one, and passing it tothe correct variable. I overcame this by carefully following the chain rule, looking at my math by hand example and verifying each step with intermediate outputs. Another thing that I noticed in my outputs is that MSE + sigmoid can get “stuck” in an very close solution. Adding some debbuging, such as looking at the true label vs prediction helped me identify this issue. 

2. Debugging is crucial to identify and fix issues that arise during training. For example, printing the predictions on the training points and seeing them all stuck around a range (often between 0.60 or 0.70) was crucial to come to my next conclusion. I belive the reason behind this behavior is that the NN got stuck in a local minima during training.

3. Using a different activation function, would change the gradient flow and saturation behavior. For example, Tanh has a range of [-1, 1], which can help with gradient flow in deeper networks. ReLU is one of my favorite activation functions as it avoids saturation issues. ReLU can indeed suffer from dead neurons. There is no silver bullet when it comes to selecting an activation function, and it is very dependant on the data. Another interesting point while using sigmoid was that it can get saturated if values get very large or very small. Leading early termination in the training.

My last point is based on the cost function used to train the network. I believe that using cross entrophy may have resulted on better results/preductions. Overall the NN performed well, and given the limited dataset.
