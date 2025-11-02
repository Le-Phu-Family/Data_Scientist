---
tags:
  - LinearRegression
  - ML
  - "#Math"
Date: 2025-09-17
Relevant:
  - "[[3.2 - Ordinary Least Square (OLS).]]"
  - "[[3.1 - Linear Regression.]]"
---

>[!important] Math convention
>By convention, vectors are typically treated as column vectors.

$$
y = \begin{bmatrix} 
y_{1} \\
y_{2} \\
\vdots \\
y_{M} 
\end{bmatrix} \in \mathbb{R}^{M \times 1} \quad \text{or} \quad \in \mathbb{R}^M
$$

## 1. RSS(w) formula.

$$
RSS(w) = \sum_{i=1}^M(y_{i} - w_{0} - w_{1}x_{i_{1}} - w_{2}x_{i_{2}} - \dots - w_{n}x_{i_{n}})^2
$$

Note:
-  The dataset $X$ is a collection of $M$ observation vectors.
	- $x_{i}$ is an observation vector in the dataset, each observation contains $n$ features.
	- After augmenting with a '1' for the bias term, $x_{i}$ becomes $x_{i} = (1, x_{i_{1}}, x_{i_{2}},\dots,x_{i_{n}})^T$, so $x_i \in \mathbb{R}^{n+1}$.
- $y_{i}$ is the **true label (scalar)** for each corresponding $x_{i}$. For instance, with observation $x_{1} = (1, 2, 3)$ we have $y_{1} = 5$.
- $w = (w_{0}, w_{1}, w_{2}, \dots, w_{n})^T$, so $w \in \mathbb{R}^{n+1}$.
	- This is the parameter vector we want to find.
- $M$ is **the number of observations/vectors** in the dataset.
- $n$ is **the number of features** in each original observation $x_{i}$ (excluding the bias term).

Define a matrix $X \in \mathbb{R}^{M \times (n+1)}$ where each **row** is an augmented observation vector $x_{i}^T$.

$$
X = \begin{pmatrix}
1 & x_{11} & x_{12} & \dots & x_{1n} \\
1 & x_{21} & x_{22} & \dots & x_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{M1} & x_{M2} & \dots & x_{Mn}
\end{pmatrix}
$$
Define a vector $y \in \mathbb{R}^{M \times 1}$ (or $\mathbb{R}^M$) which has each element as a $y_{i}$ scalar.
$$
y = \begin{bmatrix}
y_{1} \\
y_{2} \\
\vdots \\
y_{M}
\end{bmatrix}
$$
>[!question]
>Why do we have a column of '1's in matrix X?
>- To represent the bias term $w_{0} \times 1$.


## 2. Rewrite the function.
$$
RSS(w) = ||y - Xw||^2
$$
Review the [[Euclidean norm (L2 norm).]] to understand why we have this formula.

From this formula, we have the loss function below:
$$
RSS(w) = (y-Xw)^T(y-Xw)
$$
$$
\Leftrightarrow y^Ty - y^TXw - X^Tw^Ty + X^Tw^TXw
$$
$$
\Leftrightarrow y^Ty - 2w^TX^Ty+ X^Tw^TXw
$$
Our goal is to find the **minimum point** to **minimize** the $RSS(w)$ function, so we take the gradient with respect to $w$ and set it to zero.

## 3. Take the derivatives.

$$
\nabla_w RSS(w) = \frac{\partial L(w)}{\partial w}
$$
**Property 1**:
$$
\frac{\partial x^T.a}{\partial x} = a
$$

Remember from the _Math Convention_ at first, a single vector standalone is a column vector.

In the equation $2w^TX^Ty$ we have $w^T$ but we're taking the derivatives to $w$. According to property 1 we have $x^T.a$ - when multiplying with this _row vector $x^T$_ và lấy đạo hàm theo x thì ta được a.

Vậy, $-2w^TX^Ty$ lấy đạo hàm theo $w$ ta được $-2X^Ty$.

**Property 2:**
$$
\frac{\partial x^TAx}{\partial x} = (A + A^T)x
$$

In the equation $x^Tw^TXw$, we have $w^T.(X^T.X).w$ and we're taking the derivatives to $x$.
So, the result is: $((X^TX) + (X^TX)^T)w = 2X^TXw$.

### 3.1 Final gradient formula.

$$
\Leftrightarrow -2X^Ty + 2X^TXw = 0
$$
$$
\Leftrightarrow X^TXw = X^Ty
$$
$$
\Leftrightarrow w^* = (X^TX)^{-1}X^Ty
$$

We assume that $X^TX$ is invertible.

If the columns in $X$ are linearly dependent (i.e., $X$ does not have full column rank), then $X^TX$ will be non-invertible (singular).

>[!danger]
>High dimension space requires considerable computation.

