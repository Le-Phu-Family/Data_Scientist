---
tags:
  - GradientDescent
  - LinearAlgebra
  - LinearRegression
  - Math
Date: 2025-10-22
Relevant:
  - "[[Gradient Descent.]]"
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[1 - Basic Linear Regression.]]"
  - "[[3 - Linear regression - Geometric.]]"
  - "[[2 - Multiple Linear Regression.]]"
  - "[[Mathematical proof - Normal Equation.]]"
Code: Polynomial_Regression.ipynb
---
# 1. Recall.
This knowledge is from [[2 - Multiple Linear Regression.]] and [[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]].

$f(\mathbf{x};\mathbf{w})=\mathbf{w}^T \mathbf{x}=w_0+w_1 x_1+\dots+w_D x_D$

Training the model on dataset $\mathbf{D}$ =$\{(\mathbf{x}^{(i)},y^{(i)} )\}_{i=1}^N$ using the loss function:
$L(\mathbf{w})=\sum_{i=1}^N(y^{(i)}-f(\mathbf{x}^{(i) };\mathbf{w}))^2$
$=(\mathbf{y}-\mathbf{X}\mathbf{w})^T (\mathbf{y}-\mathbf{X}\mathbf{w})$
with:
$$\mathbf{X}=\begin{bmatrix}
    \mathbf{x}^{(1)T} \\
    \mathbf{x}^{(2)T} \\
    \vdots \\
    \mathbf{x}^{(N)T}
\end{bmatrix}, \quad \mathbf{y}=\begin{bmatrix}
    y^{(1)} \\
    y^{(2)} \\
    \vdots \\
    y^{(N)}
\end{bmatrix}$$

Solution ([[Mathematical proof - Normal Equation.]]):

$\mathbf{w}=(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$

# 2. Polynomial regression.

If we want to use this model:
$$
f(x, w) = w_{0} + w_{1}x + w_{2}x^2
$$
we can define:
$$
\phi(x) = \begin{bmatrix}
    \mathbf{1} \\
    \mathbf{x} \\
    \mathbf{x}^2
\end{bmatrix}
$$
We can represent $f(x,w) = \mathbf{w^t}\phi(x)$ and solve similarly to [[2 - Multiple Linear Regression.]] with $\phi(x)$ acting as the input features.

Solution ([[Mathematical proof - Normal Equation.]]):

$\mathbf{w}=(\mathbf{\phi}^T \mathbf{\phi})^{-1} \mathbf{\phi}^T \mathbf{y}$

## 2.1 Definition.

A **polynomial regression model** makes a prediction with a *polynomial function* of *input features*. With an input feature *x*, we have the following model:
$$
f(x, w) = w_{0} +w_{1}x + w_{2}x^2 + \dots + w_{D}x^D
$$
Where $\phi(x) = [1, x, x^2, \dots,x^D]^T$

**What does $D$ represent?**
In this context, $D$ represents the **degree of the polynomial**. It indicates the highest power of the input feature $x$ included in the model. For example:
*   If $D=1$, it's a linear model ($w_0 + w_1x$).
*   If $D=2$, it's a quadratic model ($w_0 + w_1x + w_2x^2$).
*   If $D=3$, it's a cubic model ($w_0 + w_1x + w_2x^2 + w_3x^3$).

**Notes:**
*   A polynomial regression model is non-linear with respect to the input features $x$.
*   A polynomial regression model is linear with respect to the weights $w$ of the model.

---

**Further explanation of "non-linear with respect to x" and "linear with respect to w":**

*   **Non-linear with respect to $x$**: This means that when you plot the graph of $f(x, w)$ with respect to $x$, you will see a curve (unless $D=1$). For example, $x^2$ or $x^3$ create curves, not straight lines.
*   **Linear with respect to $w$**:
	*   Imagine we define:
		*   $x_{new\_1} = x$
		*   $x_{new\_2} = x^2$
		*   ...
		*   $x_{new\_D} = x^D$
	*   Then, our model becomes:
		$f(x, w) = w_0 + w_1 x_{new\_1} + w_2 x_{new\_2} + \dots + w_D x_{new\_D}$
This is precisely the form of a typical multivariate linear regression model!

---
## 3. Overfitting in Polynomial Regression.

With:
- $\phi(x)$ represents input features: N x d.
- $\mathbf{w}$ represents weights: d x 1.
- $\hat{y}$ represents our prediction: N x 1.
- $y$ represents the true labels for the model to learn.
- N represents the total number of samples.
- d represents dimensions (total number of features).
For each input feature, there is a corresponding weight. Thus, for **d** input features, we aim to find a **d x 1** weight vector.

**Assumption experiment**: number of input features = number of samples (d = N).
**Training process:**
$\phi(x) = (10 \times 10)$
$\mathbf{w} = (10 \times 1)$
${y} =\phi(x).\mathbf{w} = (10\times 10)(10 \times 1) = 10 \times 1$

We are solving $y = \phi(x).\mathbf{w}$.

Since $\phi(x) = (10 \times 10)$, if it's invertible, we can directly find the optimal $\mathbf{w}$:
$$
\phi(x)^{-1}.y = \phi(x).\phi(x)^{-1}.\mathbf{w} \implies \mathbf{w} = \phi(x)^{-1}.y
$$
This will give a perfect fit on the training dataset. When making predictions:
$$
\hat{y} = \phi(x)^{-1}.\phi(x).y = I.y = y
$$
Thus, our prediction $\hat{y}$ is 100% equal to $y$ on the training set.

This will fit perfectly on the training set, but it will perform poorly on the test set.

![[Pasted image 20251023143218.png]]

In this code, we use a dataset of 10 samples and a 10th-degree polynomial feature, and the model fits perfectly on the training set.

Take a closer look.
![[Pasted image 20251023143257.png]]

Certainly, it performs poorly on the test data. The loss on the test set increased significantly compared to the training set (nearly 0).

![[Pasted image 20251023143834.png]]
