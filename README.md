# 🧠 Getting Started with Neural Networks

For this assignment, we’re taking a **first look at neural networks** — models made up of many artificial neurons. Each neuron can be seen as a **logistic regression unit**, and by combining many of them, we can create a full network of logistic regression units.

Although logistic regression isn’t the only possible building block, it’s the one that was historically used in the first neural networks and remains an intuitive one. So in this assignment, you’ll build **fully connected neural networks** with **sigmoid activations** at every layer.

The key idea is *modularity*: combining simple building blocks (modules) to form more complex systems. Many breakthroughs in AI come from **deep neural networks**, which are just very large stacks of such modules. Therefore, designing modular and reusable code is essential.

---

## 🧩 Assignment Structure

This assignment consists of **three parts**:

---

## Part 1: Modularity (Logistic Regression)

The first module you will build is **logistic regression**.  
You’ve already programmed this algorithm before, but in this assignment, you’ll modify the implementation so it fits a **modular** structure.

The mathematical core stays the same — it’s the way you’ll program and represent it that changes.

### Recap of equations

\[
\begin{aligned}
z &= \mathbf{x} \cdot \mathbf{w} + b \\
\hat{y} &= g(z) \\
g(z) &= \frac{1}{1+e^{-z}}
\end{aligned}
\]

**Cost function:**
\[
J_{\mathbf{w},b} = - \frac{1}{m} \sum_{i=1}^m y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})
\]

**Gradients:**
\[
\begin{aligned}
\frac{\partial J_{\mathbf{w},b}}{\partial b} &= \frac{1}{m}\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})\\
\frac{\partial J_{\mathbf{w},b}}{\partial w_j} &= \frac{1}{m}\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
\end{aligned}
\]

Or in vectorized form:

\[
\frac{\partial J_{\mathbf{w},b}}{\partial \mathbf{w}} = \frac{1}{m}X^T (\mathbf{\hat{y}} - \mathbf{y})
\]

You’ll still perform gradient descent to update parameters, but the key changes are:

1. **Use computational graphs** — even for simple operations like addition — so you can later handle backpropagation more easily.  
2. **Use Object-Oriented Programming (OOP)** — define each logistic regression module as a class instance that can be stacked into a larger neural network.

---

### Logistic Regression, Change 1: Computational Graphs

A convenient way to represent the mathematics in neural networks is using **computational graphs**.  
They represent how data flows through computations — especially useful for backpropagation.

For example, a simple computation \( c = a + b \) can be visualized as a graph with nodes for `a`, `b`, and `+`.  
Even complex formulas like \( c = \ln(ab + 2a^2) \) can be broken into smaller nodes.

You’ll represent **logistic regression as a computational graph**, where each operation — from the dot product to the sigmoid — is a node. Gradients will be shown as dashed arrows flowing in the reverse direction.

---

## Part 2: Building a Neural Network

In this part, you’ll take your logistic regression module and **stack multiple of them together** to form a neural network.

Each module (layer) acts as a **fully connected layer** with sigmoid activations.  
You’ll see how data flows through several connected modules — the **forward pass**.

You’ll:

- Implement `forward` and `backward` methods for your module.  
- Connect modules so that the output of one becomes the input of the next.  
- Verify that your modular implementation behaves the same as a standard feed-forward network.

By the end of this part, you’ll have a simple neural network where each layer is a modular logistic regression block.

---

## Part 3: Training a Single Module

Finally, you’ll zoom back in and **train one module** (your logistic regression) using gradient descent.

You’ll:

- Implement a **loss function** and its derivative.  
- Apply **parameter updates** to the weights and bias.  
- Observe how the module learns from data over multiple epochs.

This part introduces the **learning dynamics** you’ll later use to train an entire neural network — connecting the gradient flow you visualized earlier in the computational graph to actual learning behavior.

---

## 🧾 Summary

- Build modular versions of logistic regression using OOP.  
- Represent computations using computational graphs.  
- Stack these modules to form a neural network.  
- Train a single module to understand how learning works.  

This modular foundation will prepare you for **backpropagation and full neural network training** in the next module.
