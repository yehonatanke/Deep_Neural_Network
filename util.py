import numpy as np
import h5py
import time
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import copy

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set the default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def load_data(train_data_path, test_data_path):
    """
    Load the dataset.

    Parameters:
    - train_data_path (str): File path for the training dataset.
    - test_data_path (str): File path for the test dataset.

    Returns:
    - train_set_x_orig (numpy.ndarray): Features of the training set.
    - train_set_y_orig (numpy.ndarray): Labels of the training set.
    - test_set_x_orig (numpy.ndarray): Features of the test set.
    - test_set_y_orig (numpy.ndarray): Labels of the test set.
    - classes (numpy.ndarray): List of classes.

    Note:
    The function assumes the datasets are in HDF5 format, with specific dataset names:
    - "train_set_x" for training features
    - "train_set_y" for training labels
    - "test_set_x" for test features
    - "test_set_y" for test labels
    - "list_classes" for the list of classes

    The labels are reshaped to have shape (1, m), where m is the number of examples.
    """

    train_dataset = h5py.File(train_data_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # the train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # the train set labels

    test_dataset = h5py.File(test_data_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # the test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # the test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def show_image_and_label(index, images, labels, classes):
    """
    Display an image from the dataset along with its label.

    Arguments:
    - index (int): Index of the image to display.
    - images (numpy.ndarray): The dataset of images.
    - labels (numpy.ndarray): The labels corresponding to the images.
    - classes (numpy.ndarray): The classes representing different categories.

    Returns:
    None
    """

    plt.imshow(images[index])
    plt.show()

    label = labels[0, index]
    class_name = classes[label].decode("utf-8")

    print(f"y = {label}. It's a {class_name} picture.")


def sigmoid(Z):
    """
    Implement the sigmoid activation.

    Arguments:
    - Z (numpy.ndarray): Input array.

    Returns:
    - A (numpy.ndarray): Output of the sigmoid activation.
    - cache (numpy.ndarray): Input Z stored for computing backward pass.
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the ReLU (Rectified Linear Unit) activation.

    Arguments:
    - Z (numpy.ndarray): Input array.

    Returns:
    - A (numpy.ndarray): Output of the ReLU activation.
    - cache (numpy.ndarray): Input Z stored for computing backward pass.
    """

    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single ReLU unit.

    Arguments:
    - dA (numpy.ndarray): Post-activation gradient.
    - cache (numpy.ndarray): Input Z where we store for computing backward propagation efficiently.

    Returns:
    - dZ (numpy.ndarray): Gradient of the cost with respect to Z.
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single sigmoid unit.

    Arguments:
    - dA (numpy.ndarray): Post-activation gradient.
    - cache (numpy.ndarray): Input Z where we store for computing backward propagation efficiently.

    Returns:
    - dZ (numpy.ndarray): Gradient of the cost with respect to Z.
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(n_x, n_h, n_y):
    """
    Create and initialize the parameters of the 2-layer neural network.

    Arguments:
    - n_x (int): Size of the input layer.
    - n_h (int): Size of the hidden layer.
    - n_y (int): Size of the output layer.

    Returns:
    - parameters (dict): Parameters of the model.
                        - W1 (numpy.ndarray): Weight matrix of shape (n_h, n_x).
                        - b1 (numpy.ndarray): Bias vector of shape (n_h, 1).
                        - W2 (numpy.ndarray): Weight matrix of shape (n_y, n_h).
                        - b2 (numpy.ndarray): Bias vector of shape (n_y, 1).
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * .01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * .01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# L-layer Neural Network
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight
# matrices and bias vectors. When completing the initialize_parameters_deep function, you should make sure that
# your dimensions match between each layer. Recall that 𝑛[𝑙] is the number of units in layer 𝑙. For example,
# if the size of your input 𝑋 is (12288,209) (with 𝑚=209 examples) then:

# |            |   Shape of W         |     Shape of b  |   Activation                    |   Shape of Activation   |
# |------------|----------------------|-----------------|---------------------------------|-------------------------|
# | Layer 1    |    (𝑛[1],12288)      | (𝑛[1],1)        | 𝑍[1]=𝑊[1]𝑋+𝑏[1]                 | (𝑛[1],209)              |
# | Layer 2    |    (𝑛[2],𝑛[1])       | (𝑛[2],1)        | 𝑍[2]=𝑊[2]𝐴[1]+𝑏[2]              | (𝑛[2],209)              |
# | ⋮          | ⋮                    | ⋮               | ⋮                                | ⋮                      |
# | Layer L-1  |    (𝑛[𝐿−1],𝑛[𝐿−2])   | (𝑛[𝐿−1],1)      | 𝑍[𝐿−1]=𝑊[𝐿−1]𝐴[𝐿−2]+𝑏[𝐿−1]       | (𝑛[𝐿−1],209)           |
# | Layer L    |    (𝑛[𝐿],𝑛[𝐿−1])     | (𝑛[𝐿],1)        | 𝑍[𝐿]=𝑊[𝐿]𝐴[𝐿−1]+𝑏[𝐿]             | (𝑛[𝐿],209)             |


def initialize_parameters_deep(layer_dims):
    """
    Implement initialization for an L-layer Neural Network.

    Arguments:
    - layer_dims (list): List containing the dimensions of each layer in the network.

    Returns:
    - parameters (dict): Parameters of the model.
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * .01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    - A (numpy.ndarray): Activations from the previous layer.
    - W (numpy.ndarray): Weights matrix.
    - b (numpy.ndarray): Bias vector.

    Returns:
    - Z (numpy.ndarray): Input of the activation function.
    - cache (tuple): Tuple containing (A, W, b) for computing backward pass efficiently.
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation of the LINEAR->ACTIVATION layer.

    Arguments:
    - A_prev (numpy.ndarray): Activations from the previous layer.
    - W (numpy.ndarray): Weights matrix.
    - b (numpy.ndarray): Bias vector.
    - activation (str): Activation function to be used ("sigmoid" or "relu").

    Returns:
    - A (numpy.ndarray): Output of the activation function.
    - cache (tuple): Tuple containing linear_cache and activation_cache for computing backward pass efficiently.
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.

    Arguments:
    - X (numpy.ndarray): Input data.
    - parameters (dict): Parameters of the model.

    Returns:
    - AL (numpy.ndarray): Activation value from the output (last) layer.
    - caches (list): List of caches containing every cache of linear_activation_forward().
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function.

    Arguments:
    - AL (numpy.ndarray): Probability vector corresponding to label predictions.
    - Y (numpy.ndarray): True label vector.

    Returns:
    - cost (numpy.ndarray): Cross-entropy cost.
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g., this turns [[17]] into 17).

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Formulas:
    - 𝑑𝑊[𝑙] = ∂𝐿/∂𝑊[𝑙] = (1/𝑚)𝑑𝑍[𝑙]𝐴[𝑙−1]𝑇
    - 𝑑𝑏[𝑙] = ∂𝐿/∂𝑏[𝑙] = (1/𝑚)∑𝑖 = 1𝑚𝑑𝑍[𝑙](𝑖)
    - 𝑑𝐴[𝑙−1] = ∂𝐿/∂𝐴[𝑙−1] = 𝑊[𝑙]𝑇𝑑𝑍[𝑙]

    Arguments:
    - dZ (numpy.ndarray): Gradient of the cost with respect to the linear output.
    - cache (tuple): Tuple of values (A_prev, W, b) from forward propagation.

    Returns:
    - dA_prev (numpy.ndarray): Gradient of the cost with respect to the activation (of the previous layer).
    - dW (numpy.ndarray): Gradient of the cost with respect to W (current layer).
    - db (numpy.ndarray): Gradient of the cost with respect to b (current layer).
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    - dA (numpy.ndarray): Post-activation gradient for the current layer.
    - cache (tuple): Tuple of values (linear_cache, activation_cache).

    Returns:
    - dA_prev (numpy.ndarray): Gradient of the cost with respect to the activation (of the previous layer).
    - dW (numpy.ndarray): Gradient of the cost with respect to W (current layer).
    - db (numpy.ndarray): Gradient of the cost with respect to b (current layer).
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID group.

    Arguments:
    - AL (numpy.ndarray): Probability vector, output of the forward propagation.
    - Y (numpy.ndarray): True label vector.
    - caches (list): List of caches containing every cache of linear_activation_forward().

    Returns:
    - grads (dict): Gradients with respect to parameters.
    """

    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL

    # L'th layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # l'th layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent.

    Formulas:
    - 𝑊[𝑙] = 𝑊[𝑙]−𝛼*𝑑𝑊[𝑙]
    - 𝑏[𝑙] = 𝑏[𝑙]−𝛼*𝑑𝑏[𝑙]

    Arguments:
    - parameters (dict): Dictionary containing parameters.
    - grads (dict): Dictionary containing gradients.
    - learning_rate (float): Learning rate of the gradient descent update.

    Returns:
    - parameters (dict): Updated parameters.
    """

    parameters = copy.deepcopy(params)
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of an L-layer neural network.

    Arguments:
    - X (numpy.ndarray): Data set of examples you would like to label.
    - y (numpy.ndarray): True "label" vector.
    - parameters (dict): Parameters of the trained model.

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.

    Arguments:
    - classes (numpy.ndarray): List of classes.
    - X (numpy.ndarray): Dataset.
    - y (numpy.ndarray): True labels.
    - p (numpy.ndarray): Predicted labels.

    Returns:
    None
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # Default size of plots
    num_images = len(mislabeled_indices[0])

    num_rows = 4  # Adjust the number of rows as needed
    num_cols = int(np.ceil(num_images / num_rows))

    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))

    plt.show()


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    - X (numpy.ndarray): Input data, of shape (n_x, number of examples).
    - Y (numpy.ndarray): True "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples).
    - layers_dims (tuple): Dimensions of the layers (n_x, n_h, n_y).
    - num_iterations (int): Number of iterations of the optimization loop.
    - learning_rate (float): Learning rate of the gradient descent update rule.
    - print_cost (bool): If set to True, this will print the cost every 100 iterations.

    Returns:
    - parameters (dict): A dictionary containing W1, W2, b1, and b2.
    - costs (list): List to keep track of the cost.
    """

    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize 'parameters' dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation:
        # LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        # Compute cost
        cost = compute_cost(A2, Y)

        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        # Set grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters, costs


def plot_costs(costs, learning_rate=0.0075):
    """
    Plots the cost over iterations.

    Arguments:
    - costs (list): List containing the cost values.
    - learning_rate (float): Learning rate used in the optimization.

    Returns:
    None
    """

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    - X (numpy.ndarray): Input data, of shape (n_x, number of examples).
    - Y (numpy.ndarray): True "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples).
    - layers_dims (list): List containing the input size and each layer size, of length (number of layers + 1).
    - learning_rate (float): Learning rate of the gradient descent update rule.
    - num_iterations (int): Number of iterations of the optimization loop.
    - print_cost (bool): If True, it prints the cost every 100 steps.

    Returns:
    - parameters (dict): Parameters learnt by the model. They can then be used to predict.
    - costs (list): List to keep track of the cost.
    """

    costs = []  # keep track of cost

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    # (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)

        # Compute cost
        cost = compute_cost(AL, Y)

        # Backward propagation
        grads = L_model_backward(AL, Y, caches)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters, costs
