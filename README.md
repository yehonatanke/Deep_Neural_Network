# Deep Neural Network
A 2-layer neural network and an L-layer deep neural network to distinguish cat images from non-cat images.

## Overview

This project implements a deep neural network to distinguish between cat and non-cat images. Two models are developed:

  -  **2-layer Neural Network**
  -  **L-layer Deep Neural Network**

The models are trained and evaluated using a dataset containing cat and non-cat images. The project explores different layer configurations (𝐿 values) to enhance model performance.

## General Methodology

1. **Initialize Parameters / Define Hyperparameters**
2. **Training Loop (Num_iterations):**
    - a. Forward Propagation
    - b. Compute Cost Function
    - c. Backward Propagation
    - d. Update Parameters (using gradients from backpropagation)
3. **Use Trained Parameters to Predict Labels**

## Dataset

The dataset consists of:
- Training set with `m_train` images labeled as cat (1) or non-cat (0)
- Test set with `m_test` images labeled as 'cat' and 'non-cat'
- Each image is of shape (num_px, num_px, 3) representing RGB channels

## Usage

- Run this script in conjunction with other modules and utility files.
- Ensure that the necessary data files (e.g., train_catvnoncat.h5, test_catvnoncat.h5) are available.

## Steps:
1. **Load Dataset:**
   - train_data_path = `path/to/train_catvnoncat.h5`
   - test_data_path = `path/to/test_catvnoncat.h5`
   - train_x_orig, train_y, test_x_orig, test_y, load_data(train_data_path, test_data_path)

2. **Data Exploration**

3. **Reshape and Standardize Data**

### Two-layer Neural Network:

**The model can be summarized as: INPUT -> LINEAR -> RELU -> LINEAR -> SIGMOID -> OUTPUT.**
  - **Detailed Architecture:**
  - The input is a (64,64,3) image that is flattened to a vector of size (12288,1).
  - The corresponding vector: [𝑥0,𝑥1,...,𝑥12287]𝑇 is then multiplied by the weight matrix 𝑊[1] of size (𝑛[1],12288).
  - Then, add a bias term and take its relu to get the following vector: [𝑎[1]0,𝑎[1]1,...,𝑎[1]𝑛[1]−1]𝑇 .
  - Multiply the resulting vector by 𝑊[2] and add the intercept (bias).
  - Finally, take the sigmoid of the result. If it's greater than 0.5, classify it as a cat.

### CONSTANTS DEFINING THE MODEL ####
```python
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
```
  - The dimension of the x-axis of the input matrix: `n_x = num_px * num_px * 3`
  - The number of hidden layers: `n_h = 7`
  - The dimension of the y-axis of the input matrix: `n_y = 1`
  - The learning rate: `learning_rate = 0.0075`
  - The number of iterations: `num_iterations = 500`
  - The layers dimensions: `layers_dims = (n_x, n_h, n_y)`

***Train the Model***
```python
parameters, costs = two_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost=True)
```

***Make Predictions***
```python
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
```

### L-layer Neural Network

**The model can be summarized as: [LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID.**
  - **Detailed Architecture:**
  - The input is a (64,64,3) image that is flattened to a vector of size (12288,1).
  - The corresponding vector: [𝑥0,𝑥1,...,𝑥12287]𝑇 is then multiplied by the weight matrix 𝑊[1] Then add the intercept 𝑏[1]. The result is called the linear unit.
  - Next, the relu of the linear unit. This process could be repeated several times for each (𝑊[𝑙],𝑏[𝑙]) depending on the model architecture.
  - Finally, the sigmoid of the final linear unit. If it is greater than 0.5, it'd classify as a cat.

### CONSTANTS DEFINING THE MODEL ####
```python
n_x = 12288  # num_px * num_px * 3
layers_dims = [n_x, 20, 7, 5, 1] #  4-layer model
```

  - The dimension of the x-axis of the input matrix: `n_x = num_px * num_px * 3`
  - The input size of each layer size: `layers_dims_l = [12288, 20, 7, 5, 1] #  4-layer model`
  - The number of iterations: `num_iterations = num_iterations`

***Train the Model***
```python
parameters_l, costs2 = L_layer_model(train_x, train_y, layers_dims2, num_iterations=num_iterations, print_cost=True)
```

***Make Predictions***
```python
pred_train = predict(train_x, train_y, parameters2)
pred_test = predict(test_x, test_y, parameters2)
```

***Results Analysis***
```python
print_mislabeled_images(classes, test_x, test_y, pred_test)
```

****A few types of images the model tends to do poorly on include:****
 - Cat body in an unusual position
 - Cat appears against a background of a similar color
 - Unusual cat color and species
 - Camera Angle
 - Brightness of the picture
 - Scale variation (the cat is very large or small in image)
