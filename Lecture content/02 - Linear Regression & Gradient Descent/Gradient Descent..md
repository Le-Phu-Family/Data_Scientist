---
tags:
  - LinearRegression
  - Math
  - ML
  - "#GradientDescent"
Date: 2025-09-24
Code: Gradient_Descent.ipynb
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
---

>[!quote] Note
>This should be learned in conjunction with the code file.

## Motivation.

**Simple problems have closed-form solutions, but complex problems do not.**

-   For linear regression with one parameter → solvable manually. [[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]
-   For linear regression with multiple parameters → a closed-form solution (normal equation) still exists.
-   However, when moving to deep learning (multi-layer neural networks with millions of parameters) → there is no longer a closed-form solution, because the loss function is extremely complex, making it impossible to solve the derivative = 0 equation.
⇒ In this case, **gradient descent is the only feasible way** to find an approximate solution.

## Complex Case (multiple parameters)

However, in **real-world Linear Regression**, the model often takes the form:
**y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ**

In this scenario:
-   There are n+1 parameters to optimize: w₀, w₁, w₂, ..., wₙ
-   Partial derivatives must be calculated for each parameter: ∂L/∂w₀ = 0, ∂L/∂w₁ = 0, ...
-   This creates a **system of linear equations** with n+1 unknowns.
-   Solving this system using matrix algebra leads to the Normal Equation: **θ = (X^T X)⁻¹ X^T y**

We will not delve deeply into the concept of the **Normal Equation** for now.

---
# 1. Gradient Descent algorithm.

**Input:**
-   The function $f(x)$ to be minimized.
-   An initialized value $x_{0}$ at t = 0.
-   Learning rate $\alpha > 0$.

while [end condition not satisfied] do:
-   $g_{t}$ = $f'(x_{t})$
-   $x_{t+1} = x_{t} - \alpha g_{t}$
-   t = t + 1

From each iteration $t$, calculate the derivative at $f(x_{t})$ and update the value of $x$ for the next iteration in the opposite direction of the derivative.

>[!question]
>Why must the value $x_{t+1}$ be updated in the opposite direction of the derivative?

-   $f'(x_t) > 0$: the graph is **sloping upwards** when moving to the right → to decrease the function value, we must move **to the left** (opposite to the derivative's direction).
-   $f'(x_t) < 0$: the graph is **sloping downwards** when moving to the right → to decrease the function value, we must move **to the right** (opposite to the derivative's direction).

>[!important]
>-   The learning rate shouldn't be too large, as its step size might skip the optimal point.
>-   The learning rate shouldn't be too small, as it will take a long time to train.

---
# 2. Gradient Descent in our problem.

**Input**:
-   The loss function $L(w) = MSE(w_{0})$ to be minimized.
-   An initialized value $w_{0}$ at t = 0.
-   Learning rate $\alpha > 0$.
while [condition not satisfied] do:
-   $g_{t} = L'(w_{0})$
-   $w_{0_{t+1}} = w_{0}-\alpha.g_{t}$
-   t = t + 1
We want to find the value of $w_{0}$ that can minimize the MSE loss function.

$$
w_{0}^* = argmin_{w_{0}}L(w) = argmin_{w_{0}} \frac{1}{N} \sum_{i=1}^N(t^{(i)} - w_{0}x^{(i)})^2
$$
The derivative of the MSE loss function with respect to $w_{0}$ is:

$$
\frac{dL}{dw_{0}} = \frac{1}{N}\sum_{i=1}^N 2(w_{0}x^{(i)}-t^{(i)})x^{(i)}
$$
---
## 2.1 Dataset.

Given $D = \{(x^{(i)}, t^{(i)}) \}_{i=1}^N$:

> $x^{(i)}$ are the input features for the $i^{th}$ data point.
> $t^{(i)}$ is the target value for the $i^{th}$ data point.

In this example, target values will contain small noises $\varepsilon$.
$$
\varepsilon^{(i)} \sim N(0, \sigma^2)
$$
$$
t^{(i)} = f(x^{(i)}) + \varepsilon^{(i)}
$$
where $f(x) = 3x+2$ is an unknown target function.

**Note**: This function $f$ is used for illustration; in reality, we cannot know $f$, epsilon, or the distribution of epsilon noises.

## 2.2 Model.

Experiment with a simple model:
$$
y = f_{w}(x) = w_{0}x
$$
where $w_{0}$ is the only parameter of the model.

## 2.3 Loss function.

Using the MSE loss function to evaluate model performance.
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - f_{w}(x^{(i)}))^2
$$
## 2.4 Optimization.

Using the **Gradient Descent algorithm** to find the best $w_{0}$ value that minimizes the MSE loss function.

For each iteration $t$, we update:
$$
w_{0}^{(t+1)} = w_{0}^{(t)} - \alpha L'(w_{0}^{(t)})
$$
Here, $t$ represents the $t$-th time step/iteration. The value of $w_{0}$ at each iteration will be equal to $w_{0}$ from the previous iteration minus $\alpha$ multiplied by the derivative of $L'(w)$ at $w_{0}$ at time $t$.

![[Pasted image 20250926235137.png]]

However, the problem with a single-variable function $w_{0}x$ is that it always passes through the origin. With such sample $x^{(i)}$, no line passing through the origin will truly represent the entire dataset well.

---
## 2.5 Model.

We experiment with a new model.
$$
y = f_{w}(x) = w_{0} + w_{1}x
$$
Where $w = (w_{0}, w_{1})$ is the set of parameters to be optimized for the model.

## 2.6 Loss function.

The MSE loss function used to evaluate the model's performance will change:
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - wx^{(i)})^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
## 2.7 Optimization.

Using Gradient Descent to find the set of parameters $w = (w_{0}, w_{1})$ that minimizes the MSE loss function.

Calculate the partial derivatives of this loss function with respect to $w_{0}$ and $w_{1}$:
$$
\frac{\partial L(\mathbf{w})}{\partial w_0} = \frac{1}{N}\sum_{i=1}^N \frac{\partial}{\partial w_{0}}(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N 2((w_{0} +w_{1}x^{(i)}) - t^{(i)})
$$

$$
 \frac{\partial L(\mathbf{w})}{\partial w_{1}} = \frac{1}{N} \sum_{i=1}^N 2((w_{0} + w_{1}x^{(i)})-t^{(i)})x^{(i)}
$$

# 3. Gradient.

The **gradient $\nabla_{w}L(w)$** (also known as **partial derivatives**) of the MSE loss function $L(w)$ is defined as a vector whose components are the partial derivatives of $L(w)$ with respect to each parameter in the parameter set $w = (w_{0}, w_{1})$.

$$
\nabla_w L(\mathbf{w}) = \begin{bmatrix} \frac{\partial L(\mathbf{w})}{\partial w_0} \\ \frac{\partial L(\mathbf{w})}{\partial w_1} \end{bmatrix} = \frac{2}{N} \begin{bmatrix} \sum_{i=1}^N ((w_0 + w_1 x^{(i)}) - t^{(i)}) \cdot 1 \\ \sum_{i=1}^N ((w_0 + w_1 x^{(i)}) - t^{(i)}) \cdot x^{(i)} \end{bmatrix} = \frac{2}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{t})
$$
Notation:
$$
\mathbf{X} = \begin{bmatrix} 1 & x^{(1)} \\ 1 & x^{(2)} \\ \vdots & \vdots \\ 1 & x^{(N)} \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} t^{(1)} \\ t^{(2)} \\ \vdots \\ t^{(N)} \end{bmatrix}
$$

>[!question]
>Why is there a column of 1s in vector X?
>-   To factor out common terms: $w_{1}x^{(i)}$ factors out $x^{(i)}$, and $w_{0}$ can be seen as $w_{0} \times 1$, so 1 is factored out.

>[!question]
>Why is there $X^T$ here instead of $x^{(i)}$?

$\mathbf{X}: n \times 2$
$\mathbf{w}: 2 \times 1$
$\mathbf{X}\mathbf{w} - \mathbf{t} = \mathbf{e}: n \times 1$

If we try $\mathbf{X} \times \mathbf{e}$:

$\mathbf{X}$ is $(n \times 2)$, $\mathbf{e}$ is $(n \times 1)$ $\rightarrow$ cannot be multiplied (because the inner dimension $2 \neq n$).

However, $\mathbf{X}^T \mathbf{e}$:

$\mathbf{X}^T$ is $(2 \times n)$, $\mathbf{e}$ is $(n \times 1)$ $\rightarrow$ the result is $(2 \times 1)$, which has the same shape as $\mathbf{w}$.

>[!danger] Important
>The Gradient $\nabla_{w}L(w)$ is a vector pointing in the direction where the value of $L(w)$ increases. Since we want to minimize the loss function, we must update the parameter set $w = (w_{0}, w_{1})$ in the opposite direction of the derivative (as explained earlier): $-\nabla_{w}L(w)$.

Input:
-   The loss function $L(\mathbf{w})=\text{MSE}(w_0)$ to be minimized.
-   Initial value $\mathbf{w}$ at $t=0$.
-   Learning rate $\alpha > 0$.
while [stopping condition not satisfied] do:
		$\mathbf{g}_t = \nabla_w L(\mathbf{w})$
		$\mathbf{w} = \mathbf{w} - \alpha \mathbf{g}_t$
		$t = t+1$

## 3.1 Gradient in Gradient Descent algorithm.

Assumption: Hàm dự đoán của chúng ta là
$$
f(x) = w_{0} + w_{1}x_{1} + w_{2}x_{2} + \dots + w_{n}x_{n}
$$
And we're optimizing the loss function with respect to these parameters $(w_{0}, w_{1},\dots w_{n})$.
Tập dữ liệu có $n$ samples, each samples has the input feature $X^{(i)} = (x_{1}^{(i)}, x_{2}^{(i)}, \dots, x_{n}^{(i)})$ and their corresponding target value $y_{i}$.

Let's define predicted value is $\hat{y}$.
$$
\hat{y}^{(i)} = w_{0} + w_{1}x_{1}^{(i)} + \dots + w_{n}x_{n}^{(i)} 
$$
Loss function MSE on the whole dataset $L(w)$ is:
$$
L(w) = \frac{1}{N}\sum_{i=1}^N (\hat{y}^{(i)} - y^{(i)})^2
$$
The objective of **Gradient Descent** is to find the set of $w$ sao cho hàm mất mát này nhỏ nhất có thể.

Quy trình khởi tạo **Gradient Descent** at each iteration (For example t = 4).
1. Initialize:
	1. At the first iteration t = 0, we initialize the weights $w = w_{0}^{(0)}, w_{1}^{(1)}, \dots, w_n^{(n)}$ with random values.
2. Calculate **Gradient** at this current iteration:
	1. At t = 0, we have 



----
# 4. Learning rate.

## 4.1 Convergence.

The value of $x_{t}$ or the parameter vector $w_{t}$ will gradually approach a certain value after an infinite number of iterations ($t \to \infty$).

## 4.2 Experiment.

For the function $f(x) = 3x^2 + 4x -2$, we can calculate its derivative $f'(x) = 6x+4$.

Solving the equation $f'(x) = 0 \Leftrightarrow 6x + 4 = 0 \Leftrightarrow x = -\frac{2}{3}$.

**Note**: $x=-\frac{2}{3}$ here is where the function reaches its minimum. In reality, we would not know this point and would have to find it through the [[Gradient Descent.]] algorithm, and we also would not know the exact function $f(x)$.

So, let's implement the [[Gradient Descent.]] algorithm:
-   Choose a learning rate $\alpha > 0$.
-   Randomly initialize $x_{t}$ at step t = 1.
-   Update the value $x_{t+1}=x_{t} - \alpha.f'(x_{t})$.

Assuming we know the function $f(x)$ and its corresponding derivative $f'(x)$, at each step $t$, we update:
$x_{t+1} = x_{t}-\alpha.f'(x_{t})$
	$= x_{t} -\alpha.(6x_{t} + 4)$
	$=x_{t} - 6\alpha x_{t} - 4\alpha$
	$=x_{t}(1-6\alpha) - 4\alpha$

Let $r=(1-6\alpha)$.

**Experiment**.

_For t = 1:_
$x_{1}=rx_{0} - 4\alpha$
_For t = 2:_
$x_{2} = rx_{1}-4\alpha = r.(rx_{0}-4\alpha) - 4\alpha =r^2x_{0} - r.4\alpha -4\alpha = r^2x_{0} -4\alpha(r + 1)$
_For t = 3:_$x_{3}=rx_{2}−4α=r(r^2x_{0}−4αr−4α)−4α=r^3x_{0}−4αr^2−4α⋅r−4α =r^3x_{0}-4\alpha(r^2+r+1)$

Let $S = r^2 + r +1$. We can generalize S as a geometric series:
$S=r^{t-1} + r^{t-2}+\dots+1$
$rS = r^t + r^{t-1} + \dots + r^1$
$\implies S-rS = 1-r^t$

Solving for S:
$$
S = \frac{1-r^t}{1-r}
$$
Thus, for any $x_{t+1}$:
$$
x_{t} = r^tx_{0} - 4\alpha S
$$
With $r = 1-6\alpha \implies 1-r = 6\alpha$
$\implies x_{t} =r^t x_{0} - \frac{4\alpha.(1-r^t)}{6\alpha}=r^t x_{0}-\frac{2}{3}(1-r^t)$
$$
\implies r^t\left( x_{0}+\frac{2}{3} \right) - \frac{2}{3}
$$
For this equation to **converge** to $-\frac{2}{3}$, we need $r^t = 0 \Leftrightarrow (1-6\alpha)^t=0$. Since $x_{0}$ is an input variable that cannot be changed, and $-\frac{2}{3}$ is a constant, only the term $r^t$ (which depends on $\alpha$) can affect the expression.

$\lim_{ t \to \infty }(1-6\alpha)^t=0$ when $|1-6\alpha| <1$.

$$
-1 < 1-6\alpha < 1 \Leftrightarrow -2 < -6\alpha < 0 \Leftrightarrow \frac{-2}{-6} > \alpha > 0 \Leftrightarrow 0 < \alpha < \frac{1}{3}
$$
Therefore, for the function $f(x) =3x^2 + 4x - 2$, the **learning rate** $\alpha$ must be less than $\frac{1}{3}$ for Gradient Descent to converge to a specific value.

However, in practice, we cannot know the function $f(x)$ beforehand, so choosing a small **learning rate** $\alpha$ is always prioritized for its safety.

---
# 5. Gradient Descent & Quadratic Approximation.