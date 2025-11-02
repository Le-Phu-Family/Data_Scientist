---
tags:
  - Math
  - ML
  - GradientDescent
  - LinearRegression
Date: 2025-09-29
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Gradient Descent.]]"
  - "[[Mathematical proof - Normal Equation.]]"
  - "[[3.2 - Ordinary Least Square (OLS).]]"
  - "[[Euclidean norm (L2 norm).]]"
---
### 1. Definition of Sigma Notation

The sigma notation, $\sum$, is used to represent the sum of a series of terms.

*   **Basic Form:**
    $\sum_{i=k}^{n} a_i = a_k + a_{k+1} + a_{k+2} + \dots + a_n$
    Where:
    *   $i$: is the index of summation (the running variable).
    *   $k$: is the lower limit (the starting value of $i$).
    *   $n$: is the upper limit (the ending value of $i$).
    *   $a_i$: is the expression for the $i$-th term.

### 2. Basic Properties of Sigma Notation

These properties help simplify the calculation of complex sums.

*   **Constant Multiple Property:**
    $\sum_{i=k}^{n} c \cdot a_i = c \cdot \sum_{i=k}^{n} a_i$
    (A constant factor can be moved outside the summation sign.)

*   **Sum and Difference Property:**
    $\sum_{i=k}^{n} (a_i \pm b_i) = \sum_{i=k}^{n} a_i \pm \sum_{i=k}^{n} b_i$
    (The sum or difference of terms can be split into the sum or difference of individual sums.)

*   **Sum of a Constant:**
    $\sum_{i=1}^{n} c = n \cdot c$
    (If the lower limit is 1, the sum of a constant $c$ repeated $n$ times is $n$ times $c$.)
    **Sum of a Constant (General Form):**
    $\sum_{i=k}^{n} c = (n - k + 1) \cdot c$
    (The number of terms is $n - k + 1$.)

*   **Splitting the Sum:**
    $\sum_{i=k}^{n} a_i = \sum_{i=k}^{m} a_i + \sum_{i=m+1}^{n} a_i$ (for $k \le m < n$)
    (A sum can be divided into two or more smaller sums.)

*   **Changing the Index of Summation (Substitution):**
    $\sum_{i=k}^{n} a_i = \sum_{j=k+c}^{n+c} a_{j-c}$
    (Changing the index of summation does not change the value of the sum, as long as the term expression is adjusted accordingly.)

### 3. Common Sigma Formulas

These are formulas for specific number sequences.

*   **Sum of the first $n$ positive integers:**
    $\sum_{i=1}^{n} i = 1 + 2 + \dots + n = \frac{n(n+1)}{2}$

*   **Sum of the first $n$ squares of positive integers:**
    $\sum_{i=1}^{n} i^2 = 1^2 + 2^2 + \dots + n^2 = \frac{n(n+1)(2n+1)}{6}$

*   **Sum of the first $n$ cubes of positive integers:**
    $\sum_{i=1}^{n} i^3 = 1^3 + 2^3 + \dots + n^3 = \left(\frac{n(n+1)}{2}\right)^2$

*   **Sum of an Arithmetic Series:**
    For an arithmetic series with first term $a_1$, common difference $d$, and $n$ terms. The $i$-th term is $a_i = a_1 + (i-1)d$.
    $\sum_{i=1}^{n} a_i = \frac{n}{2}(a_1 + a_n) = \frac{n}{2}(2a_1 + (n-1)d)$

*   **Sum of a Geometric Series:**
    For a geometric series with first term $a$, common ratio $r$, and $n+1$ terms (from $i=0$ to $n$).
    $\sum_{i=0}^{n} ar^i = a + ar + ar^2 + \dots + ar^n = a \frac{1-r^{n+1}}{1-r}$ (for $r \ne 1$)
    If the sum starts from $i=1$:
    $\sum_{i=1}^{n} ar^{i-1} = a + ar + \dots + ar^{n-1} = a \frac{1-r^n}{1-r}$ (for $r \ne 1$)

*   **Sum of an Infinite Geometric Series:**
    If $|r| < 1$, the sum of an infinite geometric series converges to:
    $\sum_{i=0}^{\infty} ar^i = \frac{a}{1-r}$

### 4. Telescoping Series

A telescoping series is a series where most of the terms cancel out when summed.
*   **Basic Form:**
    $\sum_{i=k}^{n} (a_i - a_{i+1}) = (a_k - a_{k+1}) + (a_{k+1} - a_{k+2}) + \dots + (a_n - a_{n+1})$
    $= a_k - a_{n+1}$

I hope these formulas and properties are helpful to you!