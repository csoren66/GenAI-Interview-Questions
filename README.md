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

Backpropagation is a fundamental algorithm used for training artificial neural networks. It enables the network to learn by adjusting its weights and biases to minimize the error between predicted outputs and actual target values. Here's a detailed breakdown of how backpropagation works:

### 1. **Forward Pass**
- The input data is fed into the network.
- Each layer of the network processes the input by applying weights, biases, and activation functions, passing the output to the next layer until the final layer produces the output.
- The output is compared to the actual target (ground truth) to calculate the **loss** (error) using a loss function, such as mean squared error (MSE) for regression or cross-entropy for classification.

### 2. **Backward Pass (Backpropagation Step)**
- The algorithm calculates the gradient of the loss function with respect to each weight and bias in the network. This process uses the **chain rule** of calculus to propagate the error backward from the output layer to the input layer.
- Gradients represent how much the loss would change with a small change in each weight. These gradients are essential for updating the weights.

### 3. **Gradient Calculation**
- For each neuron, the algorithm computes the partial derivative of the loss with respect to its inputs, weights, and biases. This involves:
  - **Output Layer**: The gradient of the loss is calculated directly concerning the output layer's weights and biases.
  - **Hidden Layers**: The gradients are propagated backward layer by layer using the chain rule to compute how each preceding layer contributed to the overall loss.

### 4. **Weight Update**
- Once the gradients are obtained, an optimization algorithm (e.g., **stochastic gradient descent (SGD)**, **Adam**, etc.) updates the weights and biases to minimize the loss.
- The weights $w$
 are updated using the rule:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}
$$

  where $\eta$ is the **learning rate**, and $( \frac{\partial L}{\partial w} \)$ is the gradient of the loss with respect to the weight.

### 5. **Iterative Process**
- This process is repeated iteratively for multiple epochs until the network's loss converges to a minimal value, indicating that the model has learned to map inputs to outputs effectively.

### **Key Concepts Involved**
- **Chain Rule**: Essential for computing how errors are propagated through each layer.
- **Gradients**: Quantify the change needed for each weight to reduce the loss.
- **Learning Rate**: A hyperparameter that controls the step size of each weight update. A rate too high can overshoot minima; too low can slow convergence.


## 4.What is the vanishing gradient and exploding gradient problem, and how can it affect neural network training?
The **vanishing gradient** and **exploding gradient** problems are issues that occur during the training of deep neural networks, particularly those with many layers (e.g., deep feedforward or recurrent neural networks). These problems can severely impact the ability of a neural network to learn effectively. Here’s what they mean and how they affect training:

### 1. **Vanishing Gradient Problem**
- **Definition**: The vanishing gradient problem occurs when the gradients of the loss function decrease exponentially as they are propagated back through the layers of a neural network during backpropagation. As a result, the weights in the earlier layers receive extremely small updates, leading to slow learning or even stagnation (failure to learn).
- **Cause**: This often happens when activation functions such as **sigmoid** or **tanh** are used, as their derivatives are very small when the input is in certain ranges. This causes gradients to shrink as they are multiplied by these small values layer after layer.
- **Effect on Training**:
  - The network’s deeper layers learn very slowly or not at all, as their weights do not get updated effectively.
  - The model tends to rely mostly on shallow layers, leading to underfitting and poor generalization.

### 2. **Exploding Gradient Problem**
- **Definition**: The exploding gradient problem occurs when the gradients of the loss function grow exponentially during backpropagation. This leads to excessively large weight updates, causing numerical instability and making the model unable to converge.
- **Cause**: This can occur when the weights in the network have large initial values or when using poorly chosen activation functions and initialization schemes.
- **Effect on Training**:
  - The weights grow uncontrollably large, and the model can oscillate or diverge instead of converging.
  - This may cause the loss function to fluctuate dramatically or result in numerical errors (e.g., NaNs).

### **Strategies to Mitigate These Problems**
#### For Vanishing Gradients:
1. **Use Different Activation Functions**:
   - **ReLU (Rectified Linear Unit)** and its variants (e.g., **Leaky ReLU**, **Parametric ReLU**) help alleviate the vanishing gradient problem as their derivatives are constant for positive inputs and do not diminish across layers.
2. **Weight Initialization Techniques**:
   - Use initialization methods like **He initialization** for ReLU-based networks or **Xavier/Glorot initialization** for tanh-based networks to maintain appropriate variance in the activations.
3. **Batch Normalization**:
   - Normalizes the input of each layer, which helps maintain a more stable distribution of gradients, leading to faster training and reduced risk of vanishing gradients.
4. **Residual Networks (ResNets)**:
   - Introduce **skip connections** that allow gradients to flow more directly through the network, bypassing multiple layers and preventing them from vanishing.

#### For Exploding Gradients:
1. **Gradient Clipping**:
   - Clip the gradients during backpropagation to a maximum threshold to prevent them from growing too large.
2. **Use Smaller Learning Rates**:
   - Reduce the learning rate to ensure that weight updates are controlled and not excessively large.
3. **Careful Weight Initialization**:
   - Similar to preventing vanishing gradients, initializing weights using **He** or **Xavier initialization** can also prevent weights from becoming too large initially.
4. **Regularization Techniques**:
   - Apply techniques such as **L2 regularization** to keep weights within a reasonable range during training.

### **Summary**
- **Vanishing gradients** make training deep networks slow or prevent learning by causing earlier layers to receive minimal updates.
- **Exploding gradients** lead to instability and prevent convergence by causing weights to grow uncontrollably.
- Solutions involve using effective activation functions, weight initialization, normalization techniques, and gradient clipping to ensure stable and efficient training of deep networks.

## 5.How do you prevent overfitting in neural networks?
Overfitting occurs when a neural network learns the training data too well, capturing noise and details that don’t generalize to new, unseen data. This results in high accuracy on the training set but poor performance on the validation/test set. Here are effective techniques to prevent overfitting in neural networks:

### 1. **Regularization Techniques**
- **L1 and L2 Regularization**:
  - **L1 Regularization** (Lasso) adds a penalty proportional to the absolute value of the weights, promoting sparsity (many weights becoming zero).
  - **L2 Regularization** (Ridge) adds a penalty proportional to the square of the weights, preventing them from growing too large.
  - These regularizations modify the loss function by adding terms: 

$$
\text{Loss} = \text{Original Loss} + \lambda \sum_{i} w_{i}^{2} \quad (\text{for L2})
$$

  - Here, $\lambda$ is a hyperparameter controlling the regularization strength.

### 2. **Dropout**
- **Definition**: Dropout is a technique where, during training, a random fraction of neurons are temporarily “dropped out” or deactivated in each iteration. This prevents the network from becoming too reliant on any single neuron and encourages learning redundant representations.
- **Implementation**: A dropout rate (e.g., 0.5) is chosen, meaning 50% of neurons are randomly dropped during each training step.
- **Effect**: Dropout reduces overfitting and improves generalization by making the model robust to missing features.

### 3. **Early Stopping**
- **Definition**: Monitor the model’s performance on a validation set during training and stop training when the performance starts to degrade (i.e., when validation loss stops improving).
- **Effect**: Prevents the model from training too long and fitting to the noise in the training data.

### 4. **Data Augmentation**
- **Definition**: Increase the size and variability of the training data by applying transformations such as rotation, flipping, cropping, and scaling to the existing dataset.
- **Effect**: Helps the model generalize better by making it more robust to variations in the input data.

### 5. **Reduce Model Complexity**
- **Simpler Architectures**: Use smaller networks with fewer layers and neurons if possible. Large, complex networks can overfit the training data.
- **Pruning**: Remove neurons or connections that contribute less to the model’s performance to create a more compact and less overfit-prone network.

### 6. **Batch Normalization**
- **Definition**: Normalizes the inputs to each layer so that they have a mean of 0 and variance of 1. This regularization effect can reduce the need for dropout.
- **Effect**: Helps stabilize learning and can reduce overfitting by smoothing out the optimization landscape.

### 7. **Cross-Validation**
- **Definition**: Use techniques like **k-fold cross-validation** to train the model multiple times on different splits of the data and evaluate performance consistently.
- **Effect**: Provides a more robust measure of model generalization and helps tune hyperparameters.

### 8. **Ensemble Methods**
- **Definition**: Combine predictions from multiple models to form an ensemble. Common techniques include **bagging** (e.g., Random Forests) and **boosting** (e.g., Gradient Boosting).
- **Effect**: Reduces variance and improves generalization by leveraging the strengths of multiple models.

### 9. **Weight Constraints**
- **Definition**: Set a maximum limit for the weights so they do not grow beyond a certain value during training.
- **Effect**: Constrains the model’s capacity and helps prevent overfitting.

### 10. **Train with More Data**
- **Definition**: If possible, gather more training data or use **data augmentation** to simulate larger datasets.
- **Effect**: Reduces overfitting by providing the model with more diverse examples, improving its ability to generalize.

### **Summary**
To prevent overfitting in neural networks:
- Use **regularization** techniques like L1/L2.
- Apply **dropout** and **early stopping**.
- Perform **data augmentation**.
- Simplify the model and reduce complexity.
- Utilize **batch normalization**.
- Leverage **cross-validation** and **ensemble methods**.
- Constrain weights if needed and, when possible, increase the amount of training data.

Combining these strategies can help develop a model that generalizes well to unseen data.

## 6.What is dropout, and how does it help in training neural networks?
**Dropout** is a regularization technique used during the training of neural networks to reduce overfitting and improve generalization. It was introduced to prevent neural networks from becoming too reliant on specific neurons and to ensure that the model can generalize better to unseen data. Here's how dropout works and its benefits:

### 1. **How Dropout Works**
- **Random Deactivation**: During each training iteration, dropout randomly sets a fraction of the neurons (both in the input and hidden layers) to zero. This means that these neurons are effectively "dropped out" and do not participate in forward and backward passes for that iteration.
- **Dropout Rate**: The fraction of neurons to be dropped is controlled by a hyperparameter called the **dropout rate** (e.g., 0.5, meaning 50% of neurons are dropped during training). This is often denoted as \( p \).
- **Effect on Training**: Each iteration trains a different subset of the neural network, which forces the network to develop multiple independent internal representations and prevents co-adaptation of neurons.

### 2. **Why Dropout Helps**
- **Reduces Overfitting**: By randomly dropping units, dropout prevents the network from relying too heavily on specific neurons. This reduces overfitting and helps the model generalize better to new data.
- **Promotes Redundancy**: The network learns redundant representations because different neurons must handle different parts of the feature space due to dropout. This increases robustness.
- **Averaging Effect**: Dropout can be viewed as a form of ensemble learning. During testing, all neurons are active, but their weights are scaled by the dropout rate, effectively averaging the effect of different subnetworks created during training.

### 3. **Implementation Details**
- **During Training**: A mask is applied to randomly deactivate neurons at the specified dropout rate.
- **During Testing**: Dropout is not applied during testing. Instead, the outputs of neurons are scaled by \( 1 - p \) to account for the overall reduction in neuron activity during training.

### 4. **Example**
Assume a fully connected layer with 100 neurons and a dropout rate of 0.5. During each training iteration, approximately 50 of these neurons are randomly turned off. In subsequent iterations, a different set of 50 neurons may be turned off, creating different “paths” through the network each time.

### 5. **Benefits of Dropout**
- **Prevents Co-Adaptation**: By dropping out neurons, the network is forced to learn more robust features that do not rely on the presence of specific neurons.
- **Simplicity**: Dropout is easy to implement and adds minimal computational overhead during training.
- **Versatile**: Dropout can be applied to both fully connected layers and, in some cases, convolutional layers.

### **Summary**
**Dropout** is a powerful regularization technique that randomly drops neurons during training, preventing overfitting by forcing the model to develop more general and redundant internal representations. By preventing reliance on specific neurons, dropout leads to a more robust network that can generalize better to unseen data.

## 7.How do you choose the number of layers and neurons for a neural network?
Choosing the number of layers and neurons for a neural network is a crucial aspect of model design that significantly impacts its performance and generalization ability. There is no single, fixed rule for determining these parameters, but several guidelines and strategies can help you make informed decisions:

### 1. **Nature of the Problem**
- **Complexity of the Data**: If the problem involves highly complex, non-linear relationships (e.g., image recognition or natural language processing), a deeper network with more layers is typically needed. For simpler tasks (e.g., basic regression or classification), fewer layers are often sufficient.
- **Input Dimensionality**: High-dimensional data often benefits from deeper networks to extract more complex features step by step. Low-dimensional data usually requires fewer layers.

### 2. **Guidelines for Choosing the Number of Layers**
- **Start Simple**: Begin with a simple architecture (e.g., 1-2 hidden layers). If the model underfits the training data (i.e., it cannot learn the underlying patterns), then gradually increase the number of layers.
- **Empirical Testing**: Experiment with different depths to observe how performance changes. Use validation data to assess generalization and prevent overfitting.
- **Convolutional Networks**: For problems involving spatial data (e.g., images), architectures like **Convolutional Neural Networks (CNNs)** often have several convolutional layers followed by a few fully connected layers.
- **Recurrent Networks**: For sequential data (e.g., time series, language models), **Recurrent Neural Networks (RNNs)** or their variants (e.g., **LSTM**, **GRU**) are used, often with a few recurrent layers.

### 3. **Choosing the Number of Neurons per Layer**
- **Initial Approach**: Use the size of the input and output as a starting point. A common practice is to have the number of neurons in the first hidden layer be between the size of the input layer and the output layer.
- **Balanced Growth**: Start with a few neurons (e.g., similar to or slightly more than the number of input features) and scale up if the model struggles to learn effectively.
- **Incremental Tuning**: Gradually increase the number of neurons until you reach a good balance between learning capability and overfitting risk.
- **Wide vs. Deep**: A wider layer (more neurons) allows more complex representations at each layer, while a deeper network (more layers) captures hierarchical patterns. Balance between width and depth is essential for optimal performance.

### 4. **Practical Considerations**
- **Overfitting Risk**: More layers and neurons increase the capacity of the network, which can lead to overfitting. Use techniques like **dropout**, **early stopping**, and **regularization** to mitigate this risk.
- **Computational Cost**: Deeper and wider networks require more computation and memory. Make sure the model architecture fits within the available resources.
- **Model Complexity vs. Data Size**: If you have limited data, simpler models with fewer layers and neurons are preferable to avoid overfitting. Large datasets can support more complex architectures.

### 5. **Heuristics and Rules of Thumb**
- **Universal Approximation Theorem**: A single hidden layer with a sufficient number of neurons can theoretically approximate any continuous function. However, this is not practical for complex problems because it may require an impractically large number of neurons and training time.
- **Layer Sizes**:
  - **Input Layer**: Matches the number of features in the input data.
  - **Output Layer**: Matches the number of target outputs (e.g., 1 neuron for binary classification, \( n \) neurons for \( n \)-class classification).
  - **Hidden Layers**: Start with 1-2 layers and increase as needed. A common heuristic is to start with a number of neurons in the hidden layers that is a power of 2 (e.g., 32, 64, 128).

### 6. **Advanced Techniques**
- **Grid Search/Random Search**: Perform a hyperparameter search across different combinations of layer counts and neuron numbers.
- **Automated ML (AutoML)**: Use tools that automate model architecture search, such as Google's AutoML or Neural Architecture Search (NAS).
- **Transfer Learning**: For complex tasks, use pre-trained models with architectures that have been proven effective (e.g., **ResNet**, **VGG**, **Transformer models**).

### **Practical Example Workflow**
1. **Initial Network**: Start with 1-2 hidden layers and 32-128 neurons per layer.
2. **Train and Validate**: Monitor performance metrics on the validation set.
3. **Adjust Based on Results**:
   - **Underfitting**: Add more layers or increase the number of neurons.
   - **Overfitting**: Reduce the number of layers/neurons or apply regularization techniques.
4. **Hyperparameter Tuning**: Use techniques like cross-validation or hyperparameter search to refine your model.

## 8. What is transfer learning, and when is it useful?
**Transfer learning** is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second related task. This approach leverages the knowledge gained by a pre-trained model, allowing for faster training and better performance on the new task, even with limited data.

### 1. **How Transfer Learning Works**
- **Pre-Trained Model**: The process starts with a model that has been trained on a large dataset for a task that is similar to the new task. For instance, a model trained on **ImageNet** for image classification or **BERT** for natural language processing (NLP) tasks.
- **Fine-Tuning**: The pre-trained model is then adapted to the specific new task. Depending on the similarity between the original and new tasks, you can:
  - **Freeze some layers** of the pre-trained model and only train a few top layers to adapt to the new task.
  - **Fine-tune the entire model** with a lower learning rate to adjust the pre-trained weights incrementally.

### 2. **Why Transfer Learning is Useful**
- **Saves Training Time**: Training deep models from scratch is computationally intensive and time-consuming. Transfer learning provides a starting point that accelerates the training process.
- **Requires Less Data**: Transfer learning is particularly beneficial when the new task has limited labeled data. The pre-trained model already captures useful features that improve generalization.
- **Improved Performance**: Leveraging a model that has learned general features (e.g., edges, shapes in images, or sentence structures in text) often leads to better performance on the new task compared to training from scratch.

### 3. **When to Use Transfer Learning**
- **Limited Data Availability**: When you don’t have enough data to train a model from scratch.
- **Similar Tasks**: When the new task is related to the task for which the pre-trained model was originally developed (e.g., using a model trained for image classification to detect objects).
- **Complex Models**: When building complex models with many layers (e.g., CNNs, Transformer models), transfer learning helps reduce the computational resources needed.

### 4. **Common Applications of Transfer Learning**
- **Computer Vision**: Pre-trained models such as **VGG**, **ResNet**, **Inception**, and **EfficientNet** trained on large datasets like ImageNet are used as a starting point for new image classification, object detection, or segmentation tasks.
- **Natural Language Processing (NLP)**: Pre-trained language models such as **BERT**, **GPT**, and **RoBERTa** are fine-tuned for tasks like text classification, sentiment analysis, question answering, and machine translation.
- **Speech Recognition**: Transfer learning is used in models that have been trained on large speech corpora and are then adapted to specific voice commands or dialects.

### 5. **Types of Transfer Learning**
- **Feature Extraction**: Use the pre-trained model as a fixed feature extractor. Freeze the model’s layers and only train a new output layer or classifier.
- **Fine-Tuning**: Start with a pre-trained model and allow training (with a smaller learning rate) on all or part of the pre-trained layers to adapt to the new task.
- **Domain Adaptation**: Adjust the pre-trained model to work well in a domain that has different characteristics from the training domain (e.g., transferring a model trained on clean images to noisy real-world images).

### **Example of Transfer Learning in Practice**
Suppose you are developing a model for medical image classification but have a limited dataset. You can:
1. Use a pre-trained model like **ResNet** trained on ImageNet.
2. Replace the final fully connected layer to match the number of classes in your medical dataset.
3. Fine-tune the last few layers of the pre-trained model to adapt to the new dataset while keeping earlier layers frozen.

