---
tags:
  - LinearRegression
  - Math
  - ML
Date: 2025-09-24
Code: Linear_Regression.ipynb
---
>[!quote] Note
>This should be learned together with the code file.

## Dataset.

Given $D = \{(x^{(i)}, t^{(i)}) \}_{i=1}^N$:

> $x^{(i)}$ are the input features for the $i^{th}$ data point.
> $t^{(i)}$ is the target value for the $i^{th}$ data point.

**Assumption:**
$$t^{(i)} \cong f(x^{(i)})$$
Here, $f(x)$ is an **unknown function**. For example, $f(x) = 2x$ could represent the true underlying function generating the target values.

The objective is to **find or approximate this unknown function based on the given dataset**.

```python
def f(x):
    return 2 * x

np.random.seed(42)
x = np.random.uniform(0, 1, 10)
t = f(x)
```

**Note:** The function $f(x)$ in this example is merely illustrative. In reality, we cannot find the exact function $f(x^{(i)})$; we can only observe the corresponding target value $t^{(i)}$ for each data point $x^{(i)}$.

Let's experiment with a simple model of the form:
$$
y = f_{w}(x) = w_{0}x
$$
> $w_{0}$ is the model's parameter.

> The function $f(x)$ is an unknown function, and $f_{w}(x)$ is the function that needs to be approximated to $f(x)$ using the training dataset $D$.

**Objective:** To find the optimal $w_{0}$ such that $f_{w}(x)$ approximates $t^{(i)}$. This means finding a function $f_{w}(x)$ where for each input $x_{i}$, the predicted value $f_{w}(x_{i})$ closely approximates the target value $t^{(i)}$.

$t^{(i)}$ is the actual value (ground truth/true label). The model's task is to find the best function $f$ such that the predicted values approximate the actual values.

In this example [[Linear Regression.ipynb]], all data points lie on a straight line. Therefore, we can find the best $w_{0}$ by calculating the **slope**. In reality, we use [[Gradient Descent]].

$$
w_{0} = \frac{t^{(k)} - t^{(i)}}{x^{(k)} - x^{(i)}} = 2.0
$$
Slope indicates the **rate of change** of the output when the input changes by one unit:

-   If slope = 2, it means that when $x$ increases by 1 unit, $y$ increases by 2 units.
-   Positive slope: increasing trend.
-   Negative slope: decreasing trend.
-   Slope = 0: no linear relationship.

>[!important]
>Slope can only be used when:
>-   Data has a perfect linear relationship (all data points on one straight line).
>-   No noise or outliers are present.

So we have a model:
$$
f_{w}(x) = w_{0}x = 2x
$$
This model predicts all target values precisely, and this is the target function $f(x)$.

However, in reality, collected data will always have noise from measurement or observation. So, in the example above, we add a noise term $\varepsilon$ to the target value $t^{(i)}$.

**Assumption**:
$$
\varepsilon^{(i)} \sim N(0, \sigma^2)
$$
$$
t^{(i)} = f(x^{(i)}) + \varepsilon^{(i)}
$$
We assume the noise $\varepsilon^{(i)}$ follows a normal distribution. In practice, we cannot know the distribution of $\varepsilon^{(i)}$ nor can we find the exact function $f(x^{(i)})$.

In this case, with the presence of noise, we will not use the **slope** method.

Therefore, let's try different values for $w_{0}$ (as demonstrated in the Linear_Regression.ipynb notebook).

---
## Loss function.

To evaluate the model's quality, we define a loss function to measure the prediction error of the model $f_{w}(x^{(i)})$ against the target value $t^{(i)}$ for each data point $(x^{(i)}, t^{(i)})$ in the dataset $D$.

$$
L(w) = \frac{1}{N}\sum_{i = 1}^N (t^{(i)} - f_{w}(x^{(i)}))
$$
But this loss function is not good in some specific cases:
-   Some data points have negative loss values.
-   Some data points have positive loss values.
-   The sum of errors from all data points can cancel each other out, making the model appear to have good performance even if it makes many incorrect predictions.
We are only interested in the magnitude of the error rather than its sign. So, how do we eliminate the sign? We can use absolute value or squaring.

$$
L(w) = \frac{1}{N}\sum_{i = 1}^N (t^{(i)} - f_{w}(x^{(i)}))^2
$$
$$
\Leftrightarrow L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - w_{0}x^{(i)})^2
$$
So we get this function, **MSE - Mean Squared Error**.

For each $w_{0}$, we will use this formula with all input features $x_{i}$ to calculate $L(w)$ for that $w_{0}$ value.

We want to determine the optimal value of $w_{0}$ such that $L(w)$ is minimized. With this optimal $w_{0}$, we find the model $f_{w}(x) = w_{0}x$ that achieves the smallest error across the data points in dataset $D$.

$$
w^*_{0} = \underset{w_0}{\operatorname{argmin}} L(w) = \underset{w_{0}}{\operatorname{argmin}} \frac{1}{N}\sum_{i=1}^N (t^{(i)} - w_{0}x^{(i)})^2
$$
This is a quadratic function (parabola).

**To find the optimal value of parameter $w_0$, we can:**

1.  **Take the derivative** of the loss function $L(w)$ with respect to $w_0$: $\frac{dL}{dw_0}$
2.  **Set the derivative equal to zero** and solve: $L'(w_0) = 0$
3.  **The solution gives us the optimal $w_0$** that minimizes the MSE loss function.

In the model $y = w_0x$, **$w_0$ is the only parameter** for which we need to find the optimal value. The loss function $L(w_0)$ depends on $w_0$, so to find the value of $w_0$ that minimizes $L(w_0)$, we must take the derivative with respect to $w_0$.

## Basic Optimization Principle

This is a standard rule in mathematics: **to find the extremum of a function $f(x)$, we set $f'(x) = 0$**. In this case:

-   Function to optimize: $L(w_0)$
-   Optimization variable: $w_0$
-   Therefore: calculate $\frac{dL}{dw_0} = 0$

## Why Not Differentiate with Respect to Other Variables?

-   **Not differentiate with respect to $x$**: because $x$ is input data, which we cannot change.
-   **Not differentiate with respect to $t$**: because $t$ is target values, which are given.
-   **Only differentiate with respect to $w_0$**: because this is the only parameter we can adjust to minimize the loss.
