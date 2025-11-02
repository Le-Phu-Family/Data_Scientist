1. Elements of Structured Data
- Data comes from many sources: sensor measurements, events, text, images, and videos. 
- Much of this data is unstructured: 
	- images are a collection of pixels, with each pixel containing RGB (red, green, blue) color information. 
	- Texts are sequences of words and nonword char acters, often organized by sections, subsections, and so on. 
	- Clickstreams are sequences of actions by a user interacting with an app or a web page.
In fact, a major challenge of data science is to harness this torrent of raw data into actionable information.
To apply the statistical concepts covered in this book, unstructured raw data must be processed and manipulated into a structured form. One of the commonest forms of structured data is a table with rows and columns—as data might emerge from a relational database or be collected for a study.
There are two basic types of structured data: **numeric** and **categorical**
- Numeric data comes in **two forms:** 
	- **continuous**, such as wind speed or time duration, and **discrete**, such as the count of the occurrence of an event. 
	- Categorical data takes only a fixed set of values, such as a type of TV screen (plasma, LCD, LED, etc.) or a state name (Ala bama, Alaska, etc.). **Binary data** is an important special case of categorical data that takes on only one of two values, such as 0/1, yes/no, or true/false. Another useful type of categorical data is **ordinal data** in which the categories are ordered; an example of this is a numerical rating (1, 2, 3, 4, or 5)
Why do we bother with a taxonomy of data types? for the purposes of data analysis and predictive modeling, the data type is important to help determine the type of visual display, data analysis, or statistical model. More important, the data type for a variable determines how software will handle computations for that variable
![[Pasted image 20251101185755.png]]
2. Estimates Of Location
an estimate of where most of the data is located (i.e., its central tendency).
- **mean (the most basic estimate of location)**: $$_\bar{x} = \frac{\sum{x_{i}}}{n} $$
- **Trimmed mean:** dropping min and max number and then taking an average: $$\bar{x} = \frac{\sum{x_{i}}}{n - 2p} $$
	- eliminates the influence of extreme values

- **Weighted mean**: $$ {\bar{x_{w}}} = \frac{\sum{w_{i}x_{i}}}{\sum{}w_{i}} $$
	- 2 main motivations for using a weighted mean:
		- Some values are intrinsically more variable than others, and highly variable observations are given a lower weight. For example, if we are taking the average from multiple sensors and **one of the sensors is less accurate**, then **we might downweight the data from that sensor.** 
		- The data collected does not equally represent the different groups that we are interested in measuring. For example, because of the way an online experiment was conducted, we may not have a set of data that accurately reflects all groups in the user base. T**o correct that, we can give a higher weight to the values from the groups that were underrepresented.**

- median
	- The value such that one-half of the data lies above and below.
	- The median is referred to as a robust estimate of location since it is not influenced by outliers (extreme cases) that could skew the results.

```
Anomaly Detection: In contrast to typical data analysis, where outliers are sometimes informative and sometimes a nuisance, in anomaly detection the points of interest are the outliers, and the greater mass of data serves primarily to define the “normal” against which anomalies are measured.
```

- Weighted median:
	- The value such that one-half of the sum of the weights lies above and below the sorted data.
![[Pasted image 20251101213541.png]]

```
 Estimates of Location (like mean, median, mode)
These tell you where the center of your data lies.
Why important?
Summarize data simply: Instead of looking at hundreds of numbers, the mean or median gives a quick snapshot.
Compare groups: For example, comparing the average income between two cities.
Detect trends: Over time, changes in the average (e.g., average temperature or sales) can show trends.
💡 In business: Knowing the average customer spend helps with planning pricing strategies.
🔹 Estimates of Variability (like range, variance, standard deviation)
These tell you how spread out the data is.
Why important?
Understand risk/uncertainty: In finance, higher variability might mean higher risk.
Detect outliers or anomalies: If some data points are very far from the average, they may be errors or special cases.
Guide decision-making: Low variability = more consistent outcomes. High variability = need to be cautious.
💡 In machine learning: Models trained on highly variable data need to be more robust to handle uncertainty.
```

3. Estimates of Variability:

- Location is just one dimension in summarizing a feature. A second dimension, varia bility, also referred to as dispersion, measures whether the data values are tightly clus tered or spread out. At the heart of statistics lies variability: measuring it, reducing it, distinguishing random from real variability, identifying the various sources of real variability, and making decisions in the presence of it.
- **Deviations**: The difference between the observed values and the estimate of location. 
	- Synonyms **errors, residuals** 
- **Variance**: The sum of squared deviations from the mean divided by n – 1 where n is the number of data values. $$ s^2 = \frac{\sum{(x_{i}- \bar{x})^2}}{n}$$
	- Synonym mean-squared-error  **(sensitive to outliers**)
- **Standard deviation**: The square root of the variance. $$ s = \sqrt{Variance}$$
- **Mean absolute deviation:** The mean of the absolute values of the deviations from the mean.$$ \frac{\sum{|x_{i}-\bar{x}|}}{n}$$
	- Synonyms l1-norm, Manhattan norm 
- **Median absolute deviation from the median**: The median of the absolute values of the deviations from the median. 
```
it might seem peculiar that the standard deviation is preferred in statistics over the mean absolute deviation. It owes its preeminence to statistical theory: mathemati cally, working with squared values is much more convenient than absolute values, especially for statistical models.
```

```
- Các thước đo độ phân tán không hoán đổi cho nhau (do cách tính khác nhau nên thước đo khác nhau). 
- Trong phân phối chuẩn, chúng có thứ tự giá trị cố định: σ > MAD_mean > MAD_median.
- Để so sánh MAD_median với σ, người ta dùng hệ số 1.4826.
- Việc nói “50% dữ liệu nằm trong ±MAD” là cách diễn đạt trực quan: median absolute deviation mô tả phạm vi trung tâm của phân phối, nhưng phải nhân hệ số mới tương thích với σ.
```

3.1. Estimates based on Percentiles 
A different approach to estimating dispersion is based on looking at the spread of the sorted data.
- **Range**: The difference between the largest and the smallest value in a data set. **(sensitive to outliers)**
- **Order statistics**: Metrics based on the data values sorted from smallest to biggest. 
	- Synonym ranks 
- **Percentile**: The value such that P percent of the values take on this value or less and (100–P) percent take on this value or more. 
	- egs: 90th percentile của điểm thi là 85 → nghĩa là 90% học sinh có điểm ≤ 85.
	- Synonym quantile 
```
Vấn đề khi dữ liệu rất lớn
- Để tính chính xác percentile, ta thường phải **sắp xếp toàn bộ dữ liệu**.
- Với dữ liệu cực lớn (hàng triệu, hàng tỷ điểm), việc sắp xếp này rất tốn thời gian và bộ nhớ.
- Vì vậy, trong machine learning và big data, người ta dùng **thuật toán xấp xỉ** (như Zhang-Wang 2007) để tính nhanh, với sai số được đảm bảo trong một mức nhất định.

##ấn đề khi số phần tử là **chẵn**

Giả sử có nn phần tử, và ta muốn tìm percentile P.
- Nếu nn lẻ → dễ, vì ta có thể chọn đúng một phần tử trong dãy đã sắp xếp.
- Nếu nn chẵn → vị trí percentile có thể **nằm giữa hai phần tử** xjx_j và xj+1x_{j+1}.
    - Ví dụ: dữ liệu có 10 phần tử, muốn tìm median (50th percentile). Median sẽ nằm **giữa phần tử thứ 5 và thứ 6**.
        
## 4. Cách giải quyết: **nội suy tuyến tính (linear interpolation)**

- Khi percentile rơi giữa hai giá trị, ta lấy **trung bình có trọng số** (weighted average):
    

PercentileP=(1−w)⋅xj+w⋅xj+1\text{Percentile}_P = (1-w) \cdot x_j + w \cdot x_{j+1}

- Trong đó ww là một số từ 0 đến 1, xác định “mức độ gần” với xjx_j hay xj+1x_{j+1}.
    
Ví dụ:
- Dữ liệu đã sắp xếp: [2, 4, 6, 8]
- Muốn tìm 25th percentile (P=25).
- Vị trí lý thuyết = P⋅(n+1)/100=25⋅5/100=1.25P \cdot (n+1)/100 = 25 \cdot 5/100 = 1.25.
- Nghĩa là nằm giữa phần tử thứ 1 (giá trị 2) và phần tử thứ 2 (giá trị 4).
- Ta lấy: 2⋅(1−0.25)+4⋅0.25=2.52 \cdot (1-0.25) + 4 \cdot 0.25 = 2.5.
```
- **Interquartile range:** The difference between the 75th percentile and the 25th percentile. 
	- Synonym IQR


```
In statistical theory, location and variability are referred to as the first and second moments of a distribution. The third and fourth moments are called skewness and kurtosis. Skewness refers to whether the data is skewed to larger or smaller values, and kurtosis indicates the propensity of the data to have extreme values. Gener ally, metrics are not used to measure skewness and kurtosis; instead, these are discovered through visual displays.
```


## Trong Data Science / Data Analysis

- **EDA (Exploratory Data Analysis):** histogram, scatter, box plot là “must-have” để hiểu dữ liệu.
- **Feature engineering:** heatmap để xem tương quan, box plot để phát hiện outlier.
- **Model evaluation:** ROC curve, confusion matrix (categorical outcome), residual plot (numeric outcome)
✅ **Kết luận:**
- **Numeric → histogram, density, scatter, line, box/violin.**
- **Categorical → bar, stacked bar, pie (hạn chế), box/violin theo nhóm.**
- **Kết hợp numeric + categorical → grouped bar, box/violin theo nhóm, heatmap.**


# 4. Correlation
Exploratory data analysis in many modeling projects (whether in data science or in research) involves examining correlation among predictors, and between predictors X and a target variable y.
### 4.1  Pearson's correlation coefficient (sensitive to outliers): $$ r = \frac{\sum{(x_{i}-\bar{x})(y_{i}-\bar{y})}}{(n-1)s_{x}s_{y}} = \frac{COV(x,y)}{s_{x}s_{y}}$$
- assume that:
	- the relationships between X and Y is linearity
	- no outliers
	- nearly normal distribution and continuous.  
- ![[Pasted image 20251101233550.png]]
### 4.2. Spearman rank's correlation:
- instead of using real values, we use rank of the variables and calculate Pearson correlation between ranks. $$p = 1 - \frac{6\sum{[R(x_{i})-R(y_{i})]^2}}{n(n^2-1)} = 1 - \frac{6\sum{(d_{i})^2}}{n(n^2-1)}$$
	- pros: 
		- no sensitive to outliers.
		- prefers to **monotonic relationship (curved pattern)**, no linearity.
	- when: datasets have the slightly or moderately outliers or non-linear (log or exponential relationship).

### 4.3. Robust Correlation Estimators
- when: more outliers or skewness --> RCE
- MCD (Minimum Covariance Determinant):
	- idea: Tìm tập con dữ liệu có “độ phân tán nhỏ nhất” (determinant nhỏ nhất của ma trận hiệp phương sai) → nghĩa là bỏ qua những điểm outlier để ước lượng trung bình và hiệp phương sai **robust**.
	- $$r_{MCD} = \frac{COV_{MCD}(x,y)}{s_{x,MCD}s_{y,MCD}}$$
	- pros: 
		- Cực kì mạnh với dữ liệu có outliers đa chiều (multivariate outliers)
		- sử dụng sklearn.covariance

