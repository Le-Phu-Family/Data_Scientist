note: review Bias-Variance Trade-off lesson
1. Lasso regression (L1)
	- equals to 0
	- in OLS:  $$ Loss = Σ(y_{\hat{i}}ᵢ−y^​ᵢ)² $$
	- In penalty term: $$ L_{1} = Loss + λ × ∑∣βi∣      $$
	- The goals: have the ability to make coefficients of ***less important features*** to zero (Feature Selection).
	- Advantages:
		- **Avoiding overfitting** because of penalizing large coefficients. But remember that maybe reduce the accuracy of the model if lambda is not preferred.
		- **Feature selection**: automatically select most important features by penalizing the coefficients 
		- Handles large feature spaces: is effective in handling the high-dimensional data in images or videos.
	- Disadvantages:

	

2. Ridge regression (L2)
	- never to absolute 0, just shink coefficients towards 0.
	- 