# Generative AI Interview Questions

<ins>**Question on Artificial Neural Network(ANN):**</ins></br>


## 1. What is an Artificial Neural Network, and how does it work?

An **Artificial Neural Network (ANN)** is a computational model inspired by the way biological neural networks in the human brain process information. It is a key concept in the field of **machine learning** and **deep learning** and is used to solve complex problems such as image recognition, natural language processing, and more.

### Structure of an Artificial Neural Network:
1. **Neurons (Nodes)**: The basic processing units of the network. Each node represents a mathematical function that processes input and passes output to the next layer.
2. **Layers**:
   - **Input Layer**: The first layer that receives the raw input data.
   - **Hidden Layers**: Intermediate layers between the input and output layers. These layers perform transformations on the input data through weighted connections.
   - **Output Layer**: The final layer that outputs the result or prediction.

3. **Weights and Biases**:
   - **Weights** determine the strength of the connection between neurons. Each connection has an associated weight that adjusts as the network learns.
   - **Biases** help shift the activation function to better fit the data.

4. **Activation Function**:
   - Used to introduce non-linearity into the model, enabling the network to learn and represent complex patterns. Common activation functions include **ReLU (Rectified Linear Unit)**, **sigmoid**, and **tanh**.

### How Does an Artificial Neural Network Work?
1. **Forward Propagation**:
   - The input data passes through each layer of the network. Each neuron processes the data using its weights, sums it, adds a bias, and applies the activation function to produce output. This process continues until the output layer is reached.

2. **Loss Function**:
   - The network’s output is compared with the actual target (true value) using a **loss function** to calculate the error (e.g., mean squared error for regression, cross-entropy for classification).

3. **Backpropagation and Training**:
   - The **backpropagation** algorithm calculates the gradient of the loss function with respect to each weight and bias by moving backward through the network. This helps the network understand how to adjust the weights to minimize the error.
   - **Gradient Descent** (or its variants like **Stochastic Gradient Descent**) is used to update the weights and biases based on these gradients, iteratively reducing the error.

4. **Iterations**:
   - The process of forward propagation, loss computation, and backpropagation is repeated for many iterations (epochs) until the model's performance improves to a satisfactory level.

### Key Concepts in ANNs:
- **Learning Rate**: A parameter that controls how much the weights are updated during training. A learning rate that is too high can cause the model to converge too quickly to a suboptimal solution, while a rate that is too low can make training very slow.
- **Overfitting**: When the model performs well on the training data but poorly on unseen data. Techniques like **regularization**, **dropout**, and **early stopping** are used to prevent overfitting.
- **Deep Neural Networks (DNNs)**: ANNs with multiple hidden layers, capable of learning complex hierarchical patterns. These are the foundation of **deep learning**.

In summary, an **Artificial Neural Network** works by passing data through interconnected layers of nodes, adjusting weights through backpropagation and gradient descent, and learning patterns in the data to make predictions or decisions.

## 2. What are activation functions, tell me the type of the activation functions and  why are they used in neural networks?
**Activation functions** are mathematical operations used in neural networks that determine whether a neuron should be activated or not by computing a weighted sum and adding bias to decide whether it should be passed forward. They introduce non-linearity into the network, enabling it to learn and model complex data patterns.

### Why are Activation Functions Used?
- **Non-Linearity**: They introduce non-linearity to help the network learn complex patterns that a linear function alone cannot capture.
- **Output Regulation**: They control the output of neurons, shaping it to be within a specific range (e.g., 0 to 1 or -1 to 1).
- **Gradient-Based Learning**: They facilitate backpropagation by defining how errors are propagated backward during the training process.

### Types of Activation Functions

1. **Linear Activation Function**:
   - **Equation**: \( f(x) = x \)
   - **Characteristics**: Directly passes the input to the output.
   - **Limitations**: Not suitable for complex problems as it lacks non-linearity and gradient remains constant, making the network unable to learn complex data.

2. **Non-Linear Activation Functions**:
   - These allow the model to learn complex mappings from inputs to outputs. Some popular non-linear activation functions include:

   **a. Sigmoid Function**:
   - **Equation**: f(x) = 1 / (1 + e<sup>-x</sup>))
   - **Range**: (0, 1)
   - **Pros**: Good for probabilities and output layers in binary classification.
   - **Cons**: Vanishing gradient problem; gradients become very small when inputs are far from 0.

   **b. Tanh (Hyperbolic Tangent) Function**:
   - **Equation**: f(x) = (e<sup>x</sup> - e<sup>-x</sup>)/(e<sup>x</sup> + e<sup>-x</sup>)
   - **Range**: (-1, 1)
   - **Pros**: Zero-centered output helps in faster convergence.
   - **Cons**: Also suffers from vanishing gradient for extreme input values.

   **c. ReLU (Rectified Linear Unit)**:
   - **Equation**: \( f(x) = max(0, x) \)
   - **Range**: [0, ∞)
   - **Pros**: Computationally efficient and helps mitigate the vanishing gradient problem.
   - **Cons**: Can suffer from the **dying ReLU problem** where neurons get stuck during training if they output 0.

   **d. Leaky ReLU**:
   - **Equation**: f(x) = x     for x > 0
f(x) = αx    for x ≤ 0 (where α is a small positive constant)
   - **Pros**: Helps prevent dying ReLU by allowing a small, non-zero gradient for negative inputs.
   - **Cons**: Choosing the right value for α can be tricky.

   **e. Softmax Function**:
   - **Equation**: f(x<sub>i</sub>) = e<sup>x<sub>i</sub></sup> / Σ<sub>j</sub> e<sup>x<sub>j</sub></sup>    for i in {1, ..., n}
   - **Range**: (0, 1) for each output, and the sum of all outputs equals 1.
   - **Use Case**: Used in multi-class classification problems to create a probability distribution over classes.

   **f. Swish and Other Advanced Functions**:
   - **Swish**: f(x) = x · sigmoid(x) = x · (1 / (1 + e<sup>-x</sup>))
   - **Pros**: Smooth and non-monotonic, can improve training in some deep models.
   - **Cons**: Computationally more intensive than simple ReLU.

### Choosing an Activation Function
- **Hidden Layers**: ReLU and its variants (e.g., Leaky ReLU) are commonly used in hidden layers due to their computational efficiency and ability to solve the vanishing gradient problem.
- **Output Layer**:
  - **Binary Classification**: Sigmoid function is commonly used.
  - **Multi-class Classification**: Softmax function is the standard choice.
  - **Regression**: Linear activation or no activation (identity function) is often used.

Each type of activation function has unique characteristics that make it suitable for specific tasks and network architectures.
