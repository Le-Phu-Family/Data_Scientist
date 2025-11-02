Raw -> EDA -> Preprocessing -> Feature Selection 
## 1. Raw data:
goals: identify data source, shape, dtypes, summary (quick look)
- loading and summarizing:
	- info(), describe(), head()
- Classifying the types of data
	- Numeric, categorical, Datetime
- Handling missing values
	- if small, impute and if large, dropping this feature or using KNN imputer or MICE

## 2. [EDA](obsidian://open?vault=Data_Science_notes&file=Mathematics%2FExploratory%20Data%20Analysis): 
goals: understanding essence, distribution and relationship between features.
- Univariate Analysis
	- Numeric: histogram, boxplot, skewness
		--> outliers? normal distribution? 
	 - Categorical: barplot, boxplot, value_counts()
 - Bivariate Analysis:
	 - Numeric vs Numeric → scatterplot, correlation heatmap.
	- Numeric vs Categorical → violinplot, boxplot.
	- Categorical vs Categorical → crosstab, chi-square test.
- Multivariate Analysis
	- Pairplot, correlation matrix.
- Check linearity
	- scatterplot, residual.

## 3. Preprocessing
goals: normalize data 
- Handle missing values:
	- Mean/median/mode imputation (numeric)
	- Constant/Most frequent (categorical)
- Handle outliers
	- IQR, Z-score, or RobustScaler 
	- MCD for high-dimensional data
- Feature transformation
	- log/sqrt transformation (if needed,)
- Encoding 
	- LabelEncoder/OneHotEncoder
- Scaling:
	- StandardScaler (if normalize )
	- RobustScaler (if has outliers)
	- MinMaxScaler (for neural network)

## 4. Feature Analysis & Selection
- Kiểm tra multicollinearity bằng **VIF**.
- Xem tương quan giữa các biến (heatmap).
- Loại bỏ biến tương quan cao (|r| > 0.9).
-> selecting related features for modeling 