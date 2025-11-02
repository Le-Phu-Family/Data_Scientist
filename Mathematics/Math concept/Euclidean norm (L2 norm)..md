---
tags:
  - ML
  - LinearRegression
Date: 2025-09-17
Relevant:
  - "[[3.1 - Linear Regression.]]"
  - "[[3.2 - Ordinary Least Square (OLS).]]"
Source: https://fair.conf.vn/~lang/lecture/math4DS/TVLangMath4DS_F.1.pdf
---
For an n-dimensional vector 
$$
x = (x_{1}, x_{2}, \dots, x_{n})^T
$$
where $n \geq 1$, its length in its **Euclidean norm** or **2-norm** is:
$$
||x|| = \sqrt{ x_{1}^2 + x_{2}^2 + \dots + x_{n}^2 } = \sqrt{ x^T.x }
$$
We have:

$$v = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

$$\Leftrightarrow v \cdot v = \sum_{i=1}^n v_i \cdot v_i = \sum_{i=1}^n v_i^2 = v_1^2 + v_2^2 + \dots + v_n^2$$

Then we have:
$$v^T = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix}$$
$$\Leftrightarrow v^T v = \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix} \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$
$$
\Leftrightarrow v^T v = v_1 \cdot v_1 + v_2 \cdot v_2 + \dots + v_n \cdot v_n = \sum_{i=1}^n v_i^2
$$

$\implies v^T.v = v.v$, in matrix representation, we use $v^T.v$ and it's the same with $v.v$.

