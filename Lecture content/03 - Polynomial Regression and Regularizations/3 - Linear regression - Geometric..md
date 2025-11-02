---
tags:
  - GradientDescent
  - LinearRegression
  - Math
  - ML
Date: 2025-10-17
Relevant:
  - "[[Gradient Descent.]]"
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Mathematical proof - Normal Equation.]]"
Video: https://www.youtube.com/watch?v=SGbelq087Qs
---

>[!quote]
>Ignore inconsistency of notation in the picture from the Video - it's wrong.

# 1. Geometric Intepretation.

![[Pasted image 20251018160900.png]]

Let's say we have 2 features (d = 2) and 3 samples (N = 3).

The space is **3-dimensional because there are 3 samples (N=3)**.

Therefore, each **feature vector** is a 3D vector.
    
- $\mathbf{f}_1$ (height) is a 3D vector: [height_of_sample1, height_of_sample2, height_of_sample3]
- $\mathbf{f}_2$ (weight) is a 3D vector: [weight_of_sample1, weight_of_sample2, weight_of_sample3]

And accordingly, we have a 3-dimensional vector $\mathbf{Y} = [y_{1}, y_{2}, y_{3}]$.

Sketching in 3D space, we get the image as above.

![[Pasted image 20251018161100.png]]


Our model can only make predictions by combining the feature vectors: $\mathbf{\hat{Y}}$ = $w_1\mathbf{f}_1 + w_2\mathbf{f}_2$.

The set of **all possible predictions** (by trying all possible values for $w_1$ and $w_2$) forms a flat plane in our 3D space. The lecture correctly calls this a "subspace".

Randomly draw a point = $3\mathbf{f}_{1} + 2.5\mathbf{f}_{2}$ and $-5\mathbf{f}_{1} + 3\mathbf{f}_{2}$ (randomly positioned, just for simulation) $\to$ this is the linear combination of 2 vectors $\mathbf{f}_{1}, \mathbf{f}_{2}$. 

The points $3\mathbf{f}_1 + 2.5\mathbf{f}_2$ and $-5\mathbf{f}_1 + 3\mathbf{f}_2$ are just two random examples of prediction vectors that lie on this plane.

These 2 $\mathbf{f}_{1}, \mathbf{f}_{2}$ vectors create a subspace (a 2-dimensional plane).

![[Pasted image 20251018161404.png]]

And we have to find the closest point from the label to the plane. The point is on the plane so it's the linear combination of $\mathbf{f}_{1}, \mathbf{f}_{2}$ - it can 100% be represented by using 2 vectors $\mathbf{f}_{1}, \mathbf{f}_{2}$.

![[Pasted image 20251018161515.png]]

## 2. The goal.

Use the vector label ($\mathbf{Y}$ vector), project the vector $\mathbf{Y}$ orthogonally (at a $90^o$ angle) onto the subspace spanned by the feature vectors $\mathbf{f}_{1}, \mathbf{f}_{2}$ (The 2-dimensional plane).

This prediction gives us a new vector, which is our final prediction $\mathbf{\hat{Y}}$. This $\mathbf{\hat{Y}}$ is the point on the plane that is closest to $\mathbf{Y}$.

So basically we're looking for all linear combinations of feature vectors result in a point that is closest to the label.

## 3. How to understand this point?

Which one is the closest among all possible points?

Because this point is on the plane spanned by the feature vectors $\mathbf{f}_{1}, \mathbf{f}_{2}$, it surely can be represented by the linear combination of these two feature vectors.

**Define the Matrix $\mathbf{X}$:** We group our feature vectors ($\mathbf{f}_{1}, \mathbf{f}_{2}$) together as the columns of a single matrix, which we call $\mathbf{X}$.

$$
\mathbf{f}_1 = \begin{bmatrix} \text{height}_1 \\ \text{height}_2 \\ \text{height}_3 \end{bmatrix}

\mathbf{f}_2 = \begin{bmatrix} \text{weight}_1 \\ \text{weight}_2 \\ \text{weight}_3 \end{bmatrix}

\mathbf{X} = \begin{bmatrix} \begin{bmatrix} \text{height}_1 \\ \text{height}_2 \\ \text{height}_3 \end{bmatrix} & \begin{bmatrix} \text{weight}_1 \\ \text{weight}_2 \\ \text{weight}_3 \end{bmatrix} \end{bmatrix}
$$
**Define the vector $\mathbf{w}$:**
$$
\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix}
$$

$\mathbf{X}$ is (3 x 2), $\mathbf{w}$ is (2 x 1).
**Perform the Matrix-Vector Multiplication:**
$$
\mathbf{\hat{Y}} = \mathbf{Xw}
$$

![[Pasted image 20251018163126.png]]

![[Pasted image 20251018163331.png]]



| Single sample notation | Full dataset vector | Why?                                                      |
| ---------------------- | ------------------- | --------------------------------------------------------- |
| y (a scalar)           | $\mathbf{Y}$        | The $(N \times 1)$ column vector of all true labels.      |
| xᵀw (a scalar)         | $\mathbf{Xw}$       | The $(N \times 1)$ column vector of all predictions.      |
| e (a scalar)           | $\mathbf{E}$        | The $(N \times 1)$ column vector of all errors.           |
| $\hat{y}$ (a scalar)   | $\mathbf{\hat{Y}}$  | The $(N \times 1)$ column vector of all predictions.      |
| x                      | $\mathbf{X}$        | The ($N \times d$) column of vector of all input features |

Each row of $\mathbf{X}$ is a sample (1 sample has all features), and each column of $\mathbf{X}$ is a feature.

Definition:
- $\mathbf{Xw} = \mathbf{\hat{Y}}$
- $\mathbf{Y} - \mathbf{Xw} = \mathbf{E}$
- $\mathbf{E} + \mathbf{P} = \mathbf{Y}$

The geometric interpretation of linear regression is based on vector addition rule. 
- Place the tail (the start) of the error vector ($\mathbf{Y} - \mathbf{Xw}$), at the end (arrowhead) of the prediction vector $\mathbf{Xw}$.
- The sum, $\mathbf{Xw} + (\mathbf{Y} - \mathbf{Xw})$ is the new vector goes from the tail of $\mathbf{Xw}$ to the arrowhead of ($\mathbf{Y}-\mathbf{Xw}$) - vector $\mathbf{Y}$ - the label vector.

## 4. Appendix: Prove E.$\hat{Y}$ = 0.

The angel between $\mathbf{E}$ and $\mathbf{\hat{Y}}$ is $90^o$ so the dot product = 0. 
So we have, 
$$
(\mathbf{Y}-\mathbf{Xw})^T(\mathbf{Xw}) = 0 \iff \mathbf{Y}^T\mathbf{Xw} - (\mathbf{Xw})^T(\mathbf{Xw}) = 0 \iff 1 + 2 = 0
$$

Recall from [[Mathematical proof - Normal Equation.]]: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$
(1) = $(Xw^*)^T(Xw^*)$

$(Xw^*)^T(Xw^*) = (X(X^TX)^{-1}X^TY)^T(X(X^TX)^{-1}X^TY)$
$(X(X^TX)^{-1}X^TY)^T = Y^T(X^T)^T((X^TX)^{-1})^TX^T$

>[!important]
$(X^T)^T = X$
and $((X^TX)^{-1})^T = (X^TX)^{-1}$ (because the matrix is symmetric, the product $X^TX$ is always symmetric).

$(Xw^*)^T(Xw^*) = Y^TX(X^TX)^{-1}X^T X(X^TX)^{-1}X^TY$
Since $X^TX(X^TX)^{-1} = I$,
$(Xw^*)^T(Xw^*) = Y^TX(X^TX)^{-1}X^TY$.

(2) = $Y^TX.(X^TX)^{-1}.X^TY$
Since (1) = (2) so this equation = 0.

----
