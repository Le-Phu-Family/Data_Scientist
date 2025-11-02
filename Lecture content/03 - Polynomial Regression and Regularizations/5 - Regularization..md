---
tags:
  - LinearRegression
  - Math
  - ML
  - Regularization
Date: 2025-10-23
Relevant:
  - "[[1 - Basic Linear Regression.]]"
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Euclidean norm (L2 norm).]]"
  - "[[5 - Regularization.]]"
---
# 1. Ridge Regularization (L2 norm).

$$
L_{\lambda}(w) = \sum_{i=1}^N(y^{(i)} - f(x^{(i)},w)) + \lambda\sum_{k=1}^D(w_{k})^2
$$
N represents the number of samples in the dataset.
d represents dimensions (total number of features).

When starting with k = 1, we discard the $w_{0}$ from regularization. Why?

$L_{2}$ regularization uses the following formula:
$$
L_{2}=w_{1}^2 + w_{2}^2 + w_{3}^2 + \dots + w_{D}^2
$$
Regularizing w₀ would force the model to have near-zero baseline prediction, which might not make sense for the data.

**w₀** doesn't contribute to model complexity in the same way as other weights.

----
# 2. Lasso Regularization (L1 norm).

Lasso có thể đưa trọng số $w_{k}$ of some input features $x_{k}$ về giá trị 0. 
It helps making a simpler model, discarding unnecessary features.

$$
L_{\lambda}(w) = \sum_{i=1}^N(y^{(i)} - f(x^{(i)},w)) + \lambda\sum_{k=1}^D|w_{k}|
$$

