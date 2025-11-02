---
tags:
  - LinearRegression
  - GradientDescent
  - Math
  - ML
Date: 2025-09-28
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Gradient Descent.]]"
  - "[[3.1 - Linear Regression.]]"
  - "[[Sigma notation property.]]"
---
# 1. Recall.

## 1.1 Model.
Ta nhớ lại, trong bài [[Gradient Descent.]] đã giới thiệu 1 model mới là:
$$
f_{w}(x) = w_{0} + w_{1}x
$$
Với $w = (w_{0}, w_{1})$ là bộ tham số cần tối ưu hóa của model.

## 1.2 Loss function.

Hàm mất mát MSE để evaluate model's performance sẽ có thay đổi, below is _Hàm mất mát bình phương (squared loss)_, also known as **Residual of Squared Sum (RSS)**.
$$
L(w) = \sum_{1}^N(y^{(i)} - (w_{0}+ w_{1}x^{(i)}))^2
$$

We can also use this **MSE** loss function. 
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - wx^{(i)})^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
## 1.3 Optimization.

[[Sigma notation property.]] knowledge is needed to understand this.

Và tất nhiên, mục đích của chúng ta là tìm bộ tham số $w = (w_{0}, w_{1})$ sao cho khi model dùng bộ tham số này thì **Loss function** đạt giá trị nhỏ nhất - tức là, khi dùng bộ tham số này $w = (w_{0}, w_{1})$ đạt độ chính xác cao nhất.

$$\hat{w}_0, \hat{w}_1 = \underset{w_0, w_1}{\arg \min} L(w_0, w_1)$$
Tối ưu hàm $L(w_{0}, w_{1})$ theo 2 tham số $(w_{0}, w_{1})$ để tìm ra bộ $\hat{w_{0}}, \hat{w_{1}}$ tốt nhất.

Để giải quyết bài toán tối ưu hóa trên, ta có thể tính đạo hàm của $L(w)$ theo 2 biến $(w_{0}, w_{1})$ và tìm nghiệm của $\frac{\partial L(w)}{\partial w_{0}} = 0$ và $\frac{\partial L(w)}{\partial w_{1}} = 0$.

$$
L(w_{0}, w_{1}) = \sum_{i=1}^N(y^{(i)} - (w_{0} + w_{1}x))^2
$$
Partial Derivative của $L(w_{0}, w_{1})$ theo $w_{0}$ là:
$$
\frac{\partial L(w)}{\partial w_{0}}=\sum_{i=1}^N 2((w_{0} +w_{1}x^{(i)}) - t^{(i)})
$$
Ta có Partial Derivative của $L(w_{0}, w_{1})$ theo $w_{1}$ là:
$$
 \frac{\partial L(\mathbf{w})}{\partial w_{1}} =  \sum_{i=1}^N 2((w_{0} + w_{1}x^{(i)})-t^{(i)})x^{(i)}
$$

### Tìm nghiệm của 2 phương trình  $\frac{\partial L(w)}{\partial w_{0}} = 0$ và $\frac{\partial L(w)}{\partial w_{1}} = 0$.
### Giải $\frac{\partial L(w)}{\partial w_{0}}$.
$$
 \frac{\partial L(\mathbf{w})}{\partial w_{0}} = 2\sum_{1}^N(y^{(i)} - (w_{0} + w_{1}x^{(i)})).(-1)
$$
$$
= 2\sum_{1}^N((w_{0} + w_{1}x^{(i)}) - y^{(i)})
$$
### Giải $\frac{\partial L(w)}{\partial w_{0}}=0$.
$$
0 = 2\sum_{1}^N((w_{0} + w_{1}x^{(i)}) - y^{(i)})
$$
$$
0 = \sum_{1}^N((w_{0} + w_{1}x^{(i)}) - y^{(i)})
$$
Cần tìm $w_{0}$, cô lập $w_{0}$ sang 1 vế riêng:
$$
\sum w_{0} = \sum y^{(i)} - w_{1}\sum x^{(i)}
$$
$$
Nw_{0}=\sum y^{(i)} - w_{1}\sum x^{(i)}
$$
$$
w_{0} = \frac{1}{N}(\sum y^{(i)} - w_{1}\sum x^{(i)})
$$
$$
w_{0} = \bar{y} - w_{1}\bar{x}
$$

### Giải $\frac{\partial L(w)}{\partial w_{1}}$.

We have $w_{0} = \bar{y} - w_{1}\bar{x}$ from the previous step.

So we have the loss function like this:
$$
L(w) = \sum(y - (\bar{y} - w_{1}\bar{x} + w_{1}x))^2
$$
$$
= \sum(y^{(i)} - \bar{y} + w_{1}\bar{x} - w_{1}x^{(i)})^2
$$
$$
= \sum(y^{(i)} - \bar{y} + w_{1}(\bar{x} - x^{(i)}))^2
$$
$$
\frac{\partial L(w)}{\partial w_{1}} = 2\sum(y^{(i)} - \bar{y} - w_{1}(x^{(i)} - \bar{x}  ))(x^{(i)} - \bar{x})
$$
$$
=2\sum((y^{(i)} - \bar{y})(x^{(i)} - \bar{x}) - w_{1}(x^{(i)} - \bar{x})^2)
$$
### Giải $\frac{\partial L(w)}{\partial w_{1}}=0$.

$$
\sum(y^{(i)} - \bar{y})(x^{(i)} - \bar{x}) - w_{1}\sum ((x^{(i)} - \bar{x})^2) = 0
$$
$$
w_{1}\sum (x^{(i)} - \bar{x})^2 = \sum(y^{(i)} - \bar{y})(x^{(i)} - \bar{x})
$$
$$
w_{1} = \frac{\sum(y^{(i)} - \bar{y})(x^{(i)} - \bar{x})}{\sum (x^{(i)} - \bar{x})^2}
$$

**Lưu ý:** 
- $\bar{y}$ trung bình giá trị đích in dataset.
- $\bar{x}$ trung bình giá trị đặc trưng trong tập dữ liệu.
- $w_{0}, w_{1}$ is called least squared estimates.

----
