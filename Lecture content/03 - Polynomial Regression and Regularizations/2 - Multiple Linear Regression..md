---
tags:
  - LinearRegression
  - Math
  - ML
  - RSS
  - GradientDescent
Date: 2025-10-16
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Sigma notation property.]]"
  - "[[Gradient Descent.]]"
  - "[[3.1 - Linear Regression.]]"
  - "[[Euclidean norm (L2 norm).]]"
---

## 2.1 Definition.

A multiple linear regression model predict output with a linear function of input features $x_{1}, x_{2}, \dots x_{D}$.
$$f(x_1, x_2, \dots, x_D; w_0, w_1, w_2, \dots, w_D) = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_D x_D$$
$w_{0}, w_{1}, \dots, w_{D}$ is model parameterss.
$w_{0}$ is model's bias.
$w_{1}, w_{2}, \dots, w_{D}$ is model's weights, weights of each features.

$$\mathbf{w} = \begin{pmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_D \end{pmatrix} \quad \text{và} \quad \mathbf{x} = \begin{pmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_D \end{pmatrix}$$
So we have model:
$$
f(x, w) = w^Tx
$$
## 2.2 Loss function.

Loss function Squared loss of multiple linear regression model:
$$
L(w) = \sum_{1}^N(y^{(i)} - w^Tx^{(i)})^2 = \sum_{i=1}^{N} \left(y^{(i)} - \left(w_0 + \sum_{j=1}^{D} w_j x_j^{(i)}\right)\right)^2 
$$Full form:
$$
L(w) = \sum_{i=1}^{N} \left(y^{(i)} - \left(w_0 + w_1 x_1^{(i)} + w_2 x_2^{(i)} + \dots + w_D x_D^{(i)}\right)\right)^2
$$

**Lưu ý:** N is the number of samples, D is the number of input features.

## 3. Normal Equation.

[[Mathematical proof - Normal Equation.]]