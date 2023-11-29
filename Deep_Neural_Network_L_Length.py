# Deep neural network to distinguish cat images from non-cat images.
# There going to be two different models:
# - A 2-layer neural network.
# - An L-layer deep neural network.
# Evaluates the models' performance and explores diverse 𝐿 values to enhance the results.

# General Methodology:
# 1. Initialize parameters / Define hyperparameters
# 2. Loop for num_iterations:
#   - a. Forward propagation.
#   - b. Compute cost function.
#   - c. Backward propagation.
#   - d. Update parameters (using parameters, and grads from backprop).
# 3. Use trained parameters to predict labels.

from util import *

# Dataset containing:
# - a training set of `m_train` images labeled as cat (1) or non-cat (0)
# - a test set of `m_test` images labeled as 'cat' and 'non-cat'.
# - Each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

# Load dataset
train_data_path = '/Users/yehonatankeypur/Programing Projects/Python/Pycharm Projects/DeepLearningSpecialization Assignments/dataset_for_DNN/train_catvnoncat.h5'
test_data_path = '/Users/yehonatankeypur/Programing Projects/Python/Pycharm Projects/DeepLearningSpecialization Assignments/dataset_for_DNN/test_catvnoncat.h5'
train_x_orig, train_y, test_x_orig, test_y, classes = load_data(train_data_path, test_data_path)

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
print("--------------------")
print("Dataset exploration:")
print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))
print("--------------------")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # -1 makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("Reshape and standardize the data:")
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print("--------------------")

# 2-layer neural network:
# The model can be summarized as: INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT.
# Detailed Architecture:
# - The input is a (64,64,3) image that is flattened to a vector of size (12288,1).
# - The corresponding vector: [𝑥0,𝑥1,...,𝑥12287]𝑇 is then multiplied by the weight matrix 𝑊[1] of size (𝑛[1],12288).
# - Then, add a bias term and take its relu to get the following vector: [𝑎[1]0,𝑎[1]1,...,𝑎[1]𝑛[1]−1]𝑇 .
# - Multiply the resulting vector by 𝑊[2] and add the intercept (bias).
# - Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.

# L-layer neural network:
# The model can be summarized as: [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID
# Detailed Architecture:
# - The input is a (64,64,3) image that is flattened to a vector of size (12288,1).
# - The corresponding vector: [𝑥0,𝑥1,...,𝑥12287]𝑇 is then multiplied by the weight matrix 𝑊[1]
#   Then add the intercept 𝑏[1]. The result is called the linear unit.
# - Next, the relu of the linear unit. This process could be repeated several times for each (𝑊[𝑙],𝑏[𝑙]) depending on
#   the model architecture.
# - Finally, the sigmoid of the final linear unit. If it is greater than 0.5, it'd classify as a cat.

# General Methodology:
# 1. Initialize parameters / Define hyperparameters
# 2. Loop for num_iterations:
#   - a. Forward propagation.
#   - b. Compute cost function.
#   - c. Backward propagation.
#   - d. Update parameters (using parameters, and grads from backprop).
# 3. Use trained parameters to predict labels.

# 2 layer model #
# Constants to defining the model
n_x = num_px * num_px * 3  # =12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
learning_rate = 0.0075
num_iterations = 500
# Apply a two-layer neural network
print("two-layer neural network:")
parameters, costs = two_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=True)
# Use the trained parameters to classify images from the dataset
print("\n2-layer predictions:")
print("Train Accuracy:")
predictions_train = predict(train_x, train_y, parameters)
print("Test Accuracy:")
predictions_test = predict(test_x, test_y, parameters)
# Explanation:
# - The first call (predictions_train) evaluates how well the model performs on the data it was trained on.
# This gives an indication of how well the model has learned from the training dataset.
# - The second call (predictions_test) evaluates the model's performance on new, unseen data (the testing dataset).
# This provides insights into how well the model generalizes to examples it has not seen during training.
# The testing dataset serves as a proxy for how the model might perform on new, real-world data.


# L layer model
# Constants to defining the model
layers_dims_l = [12288, 20, 7, 5, 1]  # 4-layer model
print("---\nL-layer neural network:")
parameters_l, costs2 = L_layer_model(train_x, train_y, layers_dims_l, num_iterations=num_iterations, print_cost=True)
# Use the trained parameters to classify images from the dataset
print("\nL-layer predictions:")
print("Train Accuracy:")
pred_train = predict(train_x, train_y, parameters_l)
print("Test Accuracy:")
pred_test = predict(test_x, test_y, parameters_l)

# Results Analysis:
# Show images the L-layer model labeled incorrectly
print_mislabeled_images(classes, test_x, test_y, pred_test)
# A few types of images the model tends to do poorly on include:
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (the cat is very large or small in image)
