import random
import math
from .utils import scale_matrix, initialize_empty_matrix, flatten,matrix_multiplication, transpose, elementwise_substraction
from .activations import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax

class CNN():
    """
    A minimal CNN that:
        Performs a single 2D convolution with ONE filter (kernel).
        Applies ReLU activation on the convolution output.
        Flattens the activated output.
        Feeds it into a single-output FC layer witsupport for both sigmoid and 
        Uses MSE cost for training.
    """

    def __init__(self, X,Y, filter_size=(3,3),num_filters=4,epochs=100, lr=1e-2, num_classes=1):
        """
        X: list of greyscale images, each image is a 2D list (height x width).ex: X[0] is the first image.
        Y: list of scalar labels in {0,1} (binary classification).
        filter_size: type of tuple, size of the filter (height x width).
        epochs: number of training epochs.
        lr: learning rate.
        """
        self.X = X
        self.Y = Y
        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.num_filters = num_filters
        self.epochs = epochs
        self.lr = lr
        self.num_classes = num_classes

        self.img_height = len(X[0])
        self.img_width = len(X[0][0])

        # initialize the convolution filter (kernel) and FC layer parameters where FC is the fully connected layer
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        random initialization of the filter and FC layer parameters.
        convolution kernel of shape (filter_height x filter_width).
        For binary classification: weight shape (1, output_size), bias shape (1,1).
        For multi-class: weight shape (num_classes, output_size), bias shape (num_classes,1).
        """
        # stride =1 basically
        # output shape is img height - filter_height + 1 x img width - filter_width + 1
        out_height = self.img_height - self.filter_height + 1
        out_width = self.img_width - self.filter_width + 1
        self.output_size = self.num_filters * out_height * out_width

        self.conv_filters = []
        for f_idx in range(self.num_filters):
            single_filter = initialize_empty_matrix(self.filter_height, self.filter_width)
            for i in range(self.filter_height):
                for j in range(self.filter_width):
                    single_filter[i][j] = (random.random() - 0.5) * 0.2
            self.conv_filters.append(single_filter)


        if self.num_classes == 1:
            #binary classifcationm
            self.fc_weight = initialize_empty_matrix(1, self.output_size)
            for j in range(self.output_size):
                self.fc_weight[0][j] = (random.random() - 0.5) * 0.2
            self.fc_bias = initialize_empty_matrix(1,1)
            self.fc_bias[0][0] = (random.random() - 0.5) * 0.2
        else:
            #multiclass
            self.fc_weight = initialize_empty_matrix(self.num_classes, self.output_size)
            for i in range(self.num_classes):
                for j in range(self.output_size):
                    self.fc_weight[i][j] = (random.random() - 0.5) * 0.2
            self.fc_bias = initialize_empty_matrix(self.num_classes,1)
            for i in range(self.num_classes):
                self.fc_bias[i][0] = (random.random() - 0.5 )* 0.2

        # random.random() - 0.5 * 0.2 return a float in the range of 0,1
        # substarction 0.5 shifts the range to [-0.5, 0.5], centering around 0
        # multiplying by 0.2 scaled the values to [-0.1, 0.1]
        # common in neural networks becuase samll zero cented weights help rebak symmetry
        

    def _conv2D_multiple_filters(self, image):
        """
        Returns a list of shape (num_filters, out_height, out_width).
        """
        out_height = self.img_height - self.filter_height + 1
        out_width  = self.img_width  - self.filter_width  + 1

        feature_maps = []
        for f_idx in range(self.num_filters):
            filt = self.conv_filters[f_idx]
            conv_out = initialize_empty_matrix(out_height, out_width)
            for i in range(out_height):
                for j in range(out_width):
                    val = 0.0
                    for fh in range(self.filter_height):
                        for fw in range(self.filter_width):
                            val += image[i+fh][j+fw] * filt[fh][fw]
                    conv_out[i][j] = val
            feature_maps.append(conv_out)

        return feature_maps

    def _forward_pass(self, image):
        """
        forward pass over single image
        Here are the steps:
        - convolution
        - relu activation on the conv output
        - flatten the activation output
        - fully connected layer: Z = W * flat + bias.
        - Activation function:
            - sigmoid for binary classification
            - softmax for multi-class classification.
        
        this returns:
            - final_output: output after the activation
            - tuple of intermdiate values to be used during back propagation
        """
        conv_outs = self._conv2D_multiple_filters(image)

        relu_outs = []
        for f_idx in range(self.num_filters):
            relu_map = relu(conv_outs[f_idx])  
            relu_outs.append(relu_map)

        flattened_all = []
        for f_idx in range(self.num_filters):
            flat_map = flatten(relu_outs[f_idx])

            # add these onto flattened_all
            for row in flat_map:
                flattened_all.append(row)

        Z = matrix_multiplication(self.fc_weight, flattened_all)

        for i in range(len(Z)):
            Z[i][0] += self.fc_bias[i][0]
        if self.num_classes == 1:
            A = sigmoid(Z)
        else:
            A = softmax(Z)

        return A, (conv_outs, relu_outs, flattened_all, Z)
    

    def _compute_cost(self, predicted, target):
        """
        compute the cost:
        - usig MSE for binary
        - cross entrophy for multiclass
        """
        N = len(predicted)
        if self.num_classes ==1:
            sum_squared = 0.0

            for i in range(N):
                sum_squared += (predicted[i] - target[i]) **2
            return sum_squared/(2*N)
        
        else:
            total_loss = 0.0
            for i in range(N):
                # preds[i] is a 2d list (num classes x 1)
                p = initialize_empty_matrix(self.num_classes,1)

                for j in range(self.num_classes):
                    p.append(predicted[i][j][0])
                    target = [0] * self.num_classes
                    target[target[i]] = 1

                    for k in range(self.num_classes):
                        total_loss -= target[j] * math.log(p[k] + 1e-8)
            
            return total_loss / N
        
    def _conv_backprop(self, A, conv_outs, relu_outs, flattened_all, image, y):
        """
        A: final output (shape = (num_classes,1) or (1,1))
        conv_outs: list of length num_filters, each shape (out_h, out_w)
        relu_outs: likewise, each shape (out_h, out_w) after ReLU
        flattened_all: shape (num_filters*out_h*out_w, 1) 
        image: original input (32x32)
        y: label (for binary or multi-class)
        Returns: dW, dB, dFilters (list of filters gradient).
        """
        if self.num_classes == 1:
            error = A[0][0] - y
            sig_der = sigmoid_derivative(A)
            dZ_val = error * sig_der[0][0]
            dZ = [[dZ_val]]
        else:
            one_hot = [0]*self.num_classes
            one_hot[y] = 1
            dZ = initialize_empty_matrix(self.num_classes, 1)
            for i in range(self.num_classes):
                dZ[i][0] = A[i][0] - one_hot[i]
        

        dW = matrix_multiplication(dZ, transpose(flattened_all))
        dB = dZ
        
        W_T = transpose(self.fc_weight)  
        dFlat = matrix_multiplication(W_T, dZ)

        out_height = len(conv_outs[0])
        out_width  = len(conv_outs[0][0]) 
        size_per_filter = out_height * out_width

        dFilters = []
        idx_start = 0
        for f_idx in range(self.num_filters):
            chunk = []
            for c_i in range(size_per_filter):
                chunk.append(dFlat[idx_start + c_i])
            idx_start += size_per_filter

            dReluAct = initialize_empty_matrix(out_height, out_width)
            cc = 0
            for i in range(out_height):
                for j in range(out_width):
                    dReluAct[i][j] = chunk[cc][0]
                    cc += 1
            
            dConv = initialize_empty_matrix(out_height, out_width)
            relu_map = relu_outs[f_idx] 
            for i in range(out_height):
                for j in range(out_width):
                    dConv[i][j] = dReluAct[i][j]
                    if relu_map[i][j] <= 0:
                        dConv[i][j] = 0.0
            
            dFilter = initialize_empty_matrix(self.filter_height, self.filter_width)
            for i in range(out_height):
                for j in range(out_width):
                    grad_val = dConv[i][j]
                    for fh in range(self.filter_height):
                        for fw in range(self.filter_width):
                            dFilter[fh][fw] += grad_val * image[i+fh][j+fw]

            dFilters.append(dFilter)

        return dW, dB, dFilters

    def _update_parameters(self, dW, dB, dFilters):
        scaled_dW = scale_matrix(dW, self.lr)
        scaled_dB = scale_matrix(dB, self.lr)

        self.fc_weight = elementwise_substraction(self.fc_weight, scaled_dW)
        self.fc_bias   = elementwise_substraction(self.fc_bias, scaled_dB)

        for f_idx in range(self.num_filters):
            scaled_dFilter = scale_matrix(dFilters[f_idx], self.lr)
            self.conv_filters[f_idx] = elementwise_substraction(self.conv_filters[f_idx], scaled_dFilter)

    def train(self):
        N = len(self.X)

        for epoch in range(self.epochs):
            cost_sum = 0.0
            for i in range(N):
                image = self.X[i]
                label = self.Y[i]
                A, (conv_out, conv_out_act, flat, Z) = self._forward_pass(image)

                # the accumalted cost
                if self.num_classes == 1:
                    cost_sum += 0.5 * ((A[0][0] - label)**2)
                else:
                    # cross entropy
                    label = self.Y[i]
                    one_hot = [0] * self.num_classes
                    one_hot[label] = 1
                    for c in range(self.num_classes):
                        cost_sum -= one_hot[c] * math.log(A[c][0] + 1e-8)
                

                dW, dB, dFilter = self._conv_backprop(A, conv_out, conv_out_act, flat,image,label)
                self._update_parameters(dW, dB , dFilter)

            avg_cost = cost_sum / N
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Cost: {avg_cost}")


    def predict(self, image):
        """
        Predict for a single image.
        If binary => return A[0][0]
        If multi-class => return argmax
        """
        A, _ = self._forward_pass(image)
        if self.num_classes == 1:
            return A[0][0]
        else:
            best_c = 0
            best_val = A[0][0]
            for c in range(1, self.num_classes):
                if A[c][0] > best_val:
                    best_val = A[c][0]
                    best_c = c
            return best_c
