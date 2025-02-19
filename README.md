# Implimenting Deep Neural Networks with Step-Size Optimization from Scratch

## Project Overview
In this project, I implement my own custom single-layer and multi-layer neural netwrok. I explored step-size optimization techniques for deep neural network training by implementing two popular optimizers: **Stochastic Gradient Descent with Momentum (SGD+)** and **Adaptive Moment Estimation (Adam)**. Additionally, I performed **hyperparameter tuning** on the Adam optimizer’s $\beta_1$ and $\beta_2$ parameters to determine optimal values for minimizing loss after a fixed number of iterations.

## Implementation Steps

### 1. One-Neuron Classifier: Loss Function and Gradients

#### Neuron Output Formula
The output of a neuron using a **Sigmoid activation function** $g(\cdot)$ is given as:

$$
\hat{y} = g \left( \sum_{i=1}^{N} x_i a_i + b \right)
$$

where:
- $N$ is the number of input nodes.
- $a_i$ are the learnable weights associated with the input nodes.
- $b$ is the learnable bias.
- Each training sample is represented as $(x_1, x_2, \dots, x_N)$.
- $\hat{y}$is the predicted output.

#### Loss Function
For each training sample, the loss function is computed as:

$$
L = \left[ y_{\text{gt}} - g \left( \sum_{i=1}^{N} x_i a_i + b \right) \right]^2
$$

where $y_{\text{gt}}$ is the ground truth label.

#### Computing Gradients
To update the weights during training, we compute the gradient of the loss function with respect to $a_i$:

$$
\frac{\partial L}{\partial a_i} = -2 y_{\text{err}} \frac{\partial}{\partial a_i} g \left( \sum_{i=1}^{N} x_i a_i + b \right)
$$

Using the **chain rule**, with $y_{\text{err}} = y_{\text{gt}} - \hat{y}$ and $\theta = \sum_{i=1}^{N} x_i a_i + b$:

$$
\frac{\partial L}{\partial a_i} = -2 y_{\text{err}} \frac{\partial g}{\partial \theta} x_i
$$

#### Stochastic Gradient Descent (SGD) and Averaging
In **Stochastic Gradient Descent (SGD)**, we compute the loss over a batch of size $B$. Introducing an index $j$ to represent individual samples in a batch. The batch-based estimated loss is:

$$
L = \frac{1}{B} \sum_{j=1}^{B} \left[ y_{\text{gt}, j} - g \left( \sum_{i=1}^{N} x_{i,j} a_i + b \right) \right]^2
$$

##### Gradient with Respect to $a_i$
The partial derivative of the batch loss with respect to $a_i$ is:

$$
\frac{\partial L}{\partial a_i} = -\frac{2}{B} \sum_{j=1}^{B} y_{\text{err}, j} \cdot \frac{\partial g}{\partial \theta} \cdot x_{i,j}
$$

where $y_{\text{err}, j} = y_{\text{gt}, j} - \hat{y}_j$.

This gradient is then used in optimization algorithms like **SGD** or **Adam** to update the weights and minimize the loss function.


### 2. Implementing Step-Size Optimization
After setting up the baseline models, I implemented **SGD+ and Adam** as alternative optimization methods to replace the basic gradient descent.

#### **SGD with Momentum (SGD+)**
SGD+ was implemented to mitigate oscillations and accelerate convergence by introducing a momentum term:

```math
v_{t+1} = \beta v_t + g_t
```
```math
w_{t+1} = w_t - \eta v_{t+1}
```

where:
- $w_t$ represents model parameters at step $t$
- $g_t$ is the gradient of the loss function
- $v_t$ accumulates past gradients
- $\eta$ is the learning rate
- $\beta$ is the momentum coefficient (typically between 0.9 and 0.99)

#### **Adaptive Moment Estimation (Adam)**
Adam was implemented using the following update rules:

```math
m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
```
```math
v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2
```
```math
p_{t+1} = p_t - \frac{\text{lr} \cdot \hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
```

where:
- $m_t$ and $v_t$ are the first and second moment estimates of gradients
- $\beta_1, \beta_2$ are decay rates (default values: 0.9, 0.99)
- $\hat{m}$ and $\hat{v}$ are bias-corrected estimates

Adam combines **momentum and adaptive learning rates**, making it one of the most widely used optimizers in deep learning.

### 3. Performance Analysis
To assess the effectiveness of SGD+, Adam, and basic SGD, I compared:
- **Convergence speed** (time taken for loss to stabilize)
- **Final loss values**
- **Training curves** (loss vs. iterations)

Results showed that **Adam consistently outperformed basic SGD**, while **SGD+ provided a balance between fast convergence and reduced oscillations**.

### 4. Hyperparameter Tuning for Adam
I conducted an **experiment varying Adam’s $\beta_1$ and $\beta_2$ values** to analyze their impact. The following values were tested:

- $\beta_1$ ∈ {0.8, 0.95, 0.99}
- $\beta_2$ ∈ {0.89, 0.9, 0.95}

I trained the model for each combination and recorded:
- **Training time**
- **Final loss**
- **Minimum loss achieved**

From these experiments, I identified the best-performing hyperparameter values for minimizing loss while maintaining efficient training.

### 5. Visualizing Results
I generated comparative **loss vs. iteration plots** to visualize the effects of each optimizer and hyperparameter setting. These graphs highlighted:
- Faster convergence with Adam
- Improved stability with SGD+
- Performance sensitivity to $\beta_1$ and $\beta_2$ in Adam

## Data Normalization
To further optimize training, I examined the impact of **pixel value scaling** and **normalization** on model performance:

1. **Baseline normalization** – Evaluated the effect of normalizing data using a simple rescaling approach.
2. **Truncated normalization** – Implemented a more robust method by truncating input data to the interval $(\mu - 5\sigma, \mu + 5\sigma)$, where $\mu$ is the mean and $\sigma$ is the standard deviation. The truncated data was then mapped to the range $(-1.0, 1.0)$, ensuring stability in training.

Performance comparisons revealed that **truncated normalization significantly improved training consistency**.

## Key Takeaways
- **Adam outperforms SGD** due to its adaptive learning rate and momentum combination.
- **SGD+ mitigates oscillations and accelerates convergence** compared to vanilla SGD.
- **Hyperparameter tuning plays a crucial role** in optimizer performance.
- **Data normalization techniques impact model stability and convergence speed**.

## Loss for 1 Layer Network using different learning rates

<p align="center">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/one/single_out_001.jpg" width="320" title="img1">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/one/single_out_01.jpg" width="320" title="img1">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/one/single_out_1.jpg" width="320" title="img1">
</p>

## Loss for multi Layer Network using different learning rates

<p align="center">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/multi/multi_out_001.jpg" width="320" title="img1">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/multi/multi_out_01.jpg" width="320" title="img1">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/multi/multi_out_1.jpg" width="320" title="img1">
</p>

## Hyperparameter Tuning Results

<p align="center">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/one/beta_test_single.jpg" width="350" title="img1">
  <img src="https://github.com/KabirBatra06/NeuralNetwork_FromScratch/blob/main/multi/beta_test_multi.jpg" width="350" title="img1">
</p>

