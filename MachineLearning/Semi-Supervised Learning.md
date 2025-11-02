## 1. What is Semi-Supervised Learning?
- Nói một cách dễ hiểu: Bạn đang dạy cho đứa con của mình (AI) cách nhận biết hình ảnh của các loài động vật. 
	- Bạn chỉ cho nó một vài bức ảnh có ghi chú rõ ràng (labeled data). Ví dụ: "Đây là con mèo", "Đây là con chó", ...
	- Sau đó, bạn đưa cho nó một chồng các hình động vật khổng lồ khác mà không có ghi chú gì cả (unlabeled data). 
-> So, Semi-Supervised Learning is the method that helps AI learn better on both **labeled data (supervised learning)** and **unlabeled data (unsupervised learning)**.
							 **SSL = SL + UL**
## 2. Tại sao SSL lại xuất hiện & và một số điều cơ bản về SSL
- Bản chất dữ liệu trong tự nhiên (từ quá trình thu thập dữ liệu và lấy mẫu Sampling) là chủ yếu là các dữ liệu không có nhãn (unlabeled data) và thường quá trình gán nhãn (labeling) lại phải có tác động của con người (từ trực tiếp gán nhãn hay thông qua các thuật toán tự động).
- Những điều kiện có hạn bao gồm:
	- Thuận toán gán nhãn và tìm kết nối giữa đầu vào với nhãn **cần tài nguyên tính toán - khó mở rộng và kiểm soát** với kích cỡ toàn bộ dữ liệu khi tăng dần theo cấp số mũ. 
	- Tài nguyên con người có hạn và có thể lỗi và vô tình đưa thiên kiến (bias) vào bộ dữ liệu có nhãn
- Trước tình huống với số lượng dữ liệu ít ỏi dữ liệu có nhãn và số lượng lớn các dữ liệu không có nhãn - Các nhà khoa học đã đặt câu hỏi:
		**Làm thế nào để có thể huấn luyện một mô hình ML trong điều kiện ít dữ liệu có nhãn, trong khi dữ liệu "vô nhãn" lại có quá nhiều?
#### 2.1. Những điều cần lưu ý:
1.  Các thuật toán SSL được tạo nên dựa trên kết hợp 2 yếu tố UL và SL nên thường chúng cố gắng cải thiện hiệu quả thông qua việc sử dụng thông tin liên quan đến 1 trong 2 yếu tố để bổ trợ khuyết thiếu cho yếu tố còn lại.
	- Bạn có thể tưởng tượng rằng việc một đứa con lai (SSL) từ bố (SL) có khả năng kỷ luật cao nhưng thiếu sáng tạo và mẹ (UL) rất sáng tạo nhưng không kỷ luật --> đứa con là sự kết hợp hài hòa - theo khuôn khổ và sáng tạo nhưng không hề bê bối và dập khuôn.
2. SSL chỉ thực sự có ích phải phụ thuộc vào bản chất của dữ liệu và thuật toán học được sử dụng:
	 - Về bản chất dữ liệu:
		 - Dữ liệu vô nhãn được huấn luyện có chứa **thông tin** hỗ trợ trong quá trình học của máy mà dữ liệu có nhãn không có (show)
		 - hay nếu dữ liệu có nhãn có chứa **thông tin** thì khó mà có thể tách ra và "dịch" cho máy hiểu được (Show dont tell).
	- Về bản chất của thuật toán:
		- Thuật toán SSL được thiết kế để tách ra các "thông tin chìm" từ các dữ liệu vô nhãn để cung cấp kiến thức cho máy học để cải thiện hiệu suất.
		- "đây là một bài toán khó không tưởng và đầy mẫu thuẫn".
## 3. The assumptions in SSL:
- Nền tảng của SSL dựa trên những giả định cụ thể về phân phối dữ liệu đầu vào và phân phối nhãn lớp.
- Nếu các giả định được **thỏa mãn**, dữ liệu vô nhãn có thể cung cấp thông tin hữu ích giúp cải thiện độ chính xác phân loại vượt trội hơn so với việc sử dụng dữ liệu có nhãn.
-> quyết định sự thành công của SSL

Một **điều kiện cần thiết** cho học bán giám sát là **phân phối biên của dữ liệu đầu vào**, ký hiệu $p(x)$, phải chứa thông tin về **phân phối hậu nghiệm**, ký hiệu $p(y|x)$ (phân phối của nhãn khi biết đầu vào).  Nếu điều kiện này được đáp ứng, việc hiểu biết thêm về $p(x)$ thông qua dữ liệu không gán nhãn có thể giúp suy ra thêm thông tin về $p(y|x)$.

#### 3.1. Giả định tính trơn (Smoothness Assumption)
![[Pasted image 20251007003227.png]]
The smoothness assumption states that, for two input points $x, x' \in X$ that are close by in the input space, the corresponding labels $y$ and $y'$ should be the same.
Đây là một giả định phổ biến trong supervised learning, nhưng SSL mang lại lợi ích mở rộng vì có thể áp dụng chuyển tiếp qua các điểm dữ liệu trong gán nhãn.
- **Lan truyền chuyển tiếp (Transitive Propagation)**:
	- Nếu một điểm dữ liệu có nhãn ($x_1$) gần một điểm không nhãn ($x_2$), và $x_2$ lại gần một điểm không nhãn khác ($x_3$), thì ta kỳ vọng $x_3$ sẽ có cùng nhãn với $x_1$, dù $x_1$ và $x_3$ không trực tiếp gần nhau. Nhãn được **truyền gián tiếp** thông qua điểm trung gian $x_2$.
	- Tóm lại, $x_1$ gần $x_{2}$ và $x_{2}$ lại gần unlabeled data point $x_{3}$ thì ta **có thể** suy ra rằng $x_1$ và $x_3$ cùng nhãn dù chúng không gần nhau. 

#### 3.2. Giả định Mật độ thấp (Low-Density Assumption)
Giả định mật độ thấp phát biểu rằng **ranh giới quyết định (decision boundary) của bộ phân loại nên đi qua các vùng có mật độ dữ liệu thấp trong không gian đầu vào**.  (Coi hình trên)
Nói cách khác, **ranh giới không nên cắt qua những vùng mà phân phối dữ liệu thực tế $p(x)$ có giá trị cao.** 
- Mối liên hệ với smoothness assumption:
	- Nếu ranh giới quyết định nằm trong **vùng mật độ cao**, điều đó có nghĩa là **các điểm dữ liệu tương tự nhau (trong vùng dày đặc)** lại bị gán **nhãn khác nhau**, vi phạm giả định tính trơn.
	- Ngược lại, nếu giả định tính trơn đúng, **các điểm trong vùng mật độ cao** sẽ có **cùng nhãn**, khiến **ranh giới quyết định chỉ có thể nằm ở vùng mật độ thấp**.

#### 3.3. Giả định Đa tạp (Manifold Assumption)
![[Pasted image 20251007011010.png]]
Giả định đa tạp đề cập đến tình huống phổ biến trong các bài toán học máy có không gian đầu vào nhiều chiều, nơi **các điểm dữ liệu quan sát được tập trung trên những cấu trúc con có số chiều thấp hơn**, gọi là **đa tạp (manifold)**.
Giả định này gồm hai phần:
1. Không gian đầu vào bao gồm nhiều **đa tạp có số chiều thấp hơn**, nơi các điểm dữ liệu cùng nằm trên đó.
2. **Các điểm nằm trên cùng một đa tạp được kỳ vọng có cùng nhãn.**
- **Suy luận:**  
    Nếu có thể xác định được các đa tạp, thì **các điểm dữ liệu không gán nhãn** có thể được **suy ra nhãn từ các điểm có nhãn nằm trên cùng đa tạp đó**.

- **Liên hệ với phương pháp:**
    - Giả định này là cơ sở của **nhiều thuật toán bán giám sát nội tại**, đặc biệt là **các phương pháp dựa trên đồ thị** (cả quy nạp và suy diễn).
    - Các phương pháp này **xấp xỉ cấu trúc đa tạp** bằng cách **xây dựng đồ thị dựa trên độ tương đồng cục bộ** giữa các điểm dữ liệu.
    - **Kỹ thuật điều chuẩn đa tạp (Manifold Regularization)** định nghĩa đồ thị và **phạt các sai khác trong dự đoán** của những điểm có khoảng cách địa lý (geodesic distance) nhỏ trên đồ thị đó, qua đó **ngầm thực thi giả định đa tạp**
    - **Autoencoder**, được dùng trong giai đoạn **tiền xử lý không giám sát**, cũng dựa trên giả định rằng **các mẫu nằm trên cùng một cấu trúc con có số chiều thấp hơn sẽ có cùng nhãn.**

#### 3.3. Giả định Phân cụm (Cluster Assumption) - Tổng quát hóa
- phát biểu rằng **các điểm dữ liệu thuộc cùng một cụm thì thuộc cùng một lớp**
- **Mối liên hệ với khái niệm tương tự (similarity):**  
    Phân cụm dựa trên ý tưởng rằng **các điểm trong cùng một cụm giống nhau nhiều hơn so với các điểm ngoài cụm**.  
    Ba giả định SSL chính có thể được xem như **những cách định nghĩa khác nhau về “sự giống nhau”**, dẫn đến giả định phân cụm:
    - **Giả định tính trơn:** Các điểm **gần nhau trong không gian đầu vào** được coi là tương tự.
    - **Giả định mật độ thấp:** Các điểm **trong cùng vùng mật độ cao** được coi là tương tự.
    - **Giả định đa tạp:** Các điểm **nằm trên cùng một đa tạp** được coi là tương tự.
- **Điều kiện nền tảng:**  
    Giả định phân cụm thậm chí có thể được xem là **điều kiện cần thiết để học bán giám sát có thể thành công** — nếu dữ liệu **không thể phân cụm một cách có ý nghĩa**, thì **học bán giám sát sẽ không thể cải thiện so với học có giám sát.**

## 4. Inductive and Transductive Learning (2 kiểu học cao nhất với bài toán "Show don't tell")
![[Pasted image 20251007181543.png]]

| Loại phương pháp                      | Đầu ra/ Mục tiêu chính                                                                                                                                                                              | Trọng tâm tối ưu hóa                                         |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Phương pháp Suy diễn (Inductive)**  | Tạo ra một mô hình phân loại ($f: \mathcal {X} \mapsto \mathcal {Y}$) có thể dự đoán nhãn cho _bất kỳ_ đối tượng nào trong không gian đầu vào, bao gồm cả các điểm dữ liệu chưa từng thấy trước đó. | Tối ưu hóa trên các mô hình dự đoán.                         |
| **Phương pháp Duy lý (Transductive)** | Chỉ tạo ra các nhãn dự đoán ($\hat{\mathbf {y}}_U$) cho **tập dữ liệu chưa gán nhãn ($X_U$)** được gặp trong giai đoạn huấn luyện.                                                                  | Tối ưu hóa trực tiếp trên các dự đoán $\hat{\mathbf {y}}_U$. |
```
“Khi giải quyết một vấn đề cụ thể, đừng giải quyết một vấn đề tổng quát hơn như là một bước trung gian. Hãy cố gắng tìm ra câu trả lời mà bạn thực sự cần chứ không phải một câu trả lời tổng quát hơn.” - Vladimir Vapnik (source: Wikipedia)
```


## 4. Wrapper Method's - "Tối cổ nhưng lợi hại" -- Pseudo-labeling
Đây là method "tối cổ nhất", nhưng vẫn đang sử dụng thịnh hành trong các bài toán phân loại bằng ý tưởng SSL.
- Ý tưởng của Wrapper Method là bạn có n chiếc máy phân loại (n classifiers) - tất cả chưa được học gì cả
- Từng máy trong n máy được huấn luyện trên 2 loại dữ liệu:
	- Dữ liệu gốc đã được gán nhãn - số lượng rất ít
	- Dữ liệu vô nhãn, nhưng tất cả sẽ được gán cho một "bí danh - alias" tạm thời - pseudo labels, và cái bí danh này sẽ luôn được cập nhật bằng mô hình sau lần học mới nhất (latest training iteration)
- Dựa vào 2 kiểu dữ liệu (dữ liệu với tên thật, và dữ liệu dùng bí danh) - thì Wrapper Method cũng có 2 bước trong thiết kế một mô hình phân loại theo ý tưởng SSL:
    - **Bước 1**: Training (Huấn luyện) - máy phân loại được học bằng dữ liệu mang tên thật và dữ liệu mang “Alias”
    - **Bước 2**: Pseudo-Labeling (Cập nhập bí danh) - máy phân loại sẽ dùng lượng kiến thức mà nó đã được học mới nhất để sửa các “Alias” cho các dữ liệu “vô nhãn”, và để bước vào lần huấn luyện mới.
![[Pasted image 20251007183354.png]]

## 5. Self-traning, Co-training and Classification problems:
Ở phần trên mình nói là phương pháp Wrapper có sử dụng đến n - máy phân loại (Classifier"), nhưng **không có nói rõ n** là một giá trị như thế nào, và đây chính là mấu chốt cho 2 kiểu vấn đề con (sub-problem) của Wrapper Method: **Self-training và Co-training**.

#### 5.1. Self-training (self-learning):
là phương pháp để thực hiện Pseudo-labeling đơn giản nhất - chỉ cần dùng một máy phân loại (n=1) vừa học vừa gán nhãn giả cho dữ liệu không mang nhãn.
Quy trình:
- Quá trình này bắt đầu huấn luyện với toàn bộ dữ liệu có nhãn
- Sau đó thu được mô hình - và thử dùng với dữ liệu không nhãn -> chọn trong số dữ liệu được gán "bí danh" với tự tin là chính xác cao nhất để thêm vào tập dữ liệu có nhãn, những điểm còn lại không được chọn bị xóa tên và để gán lần tiếp theo.
- Lặp lại bước trên cho đến khi tập dữ liệu không được gán nhãn không còn điểm nào hoặc khi mô hình hội tụ.

##### Các yếu tố quan trọng:
hiệu quả của self-training phụ thuộc mạnh vào một số quyết định:
- Quy trình chọn mẫu: Việc chọn điểm dữ liệu nào để gán nhãn giả là yếu tố then chốt --> quyết định **chất lượng** của tập dữ liệu mở rộng --> thông thường dựa vào **mức độ tự tin** trong dự đoán.
- Chất lượng độ tin cậy: Mức xác suất dự đoán phải phán ánh đúng độ tự tin thực tế của mô hình.
- Giả định của mô hình cơ sở: Các giả định cốt lõi của SSL trong self-training **phụ thuộc hoàn toàn vào mô hình cơ sở** được sử dụng. Ví dụ, khi kết hợp self-training với SVM, mô hình ngầm giả định **“low-density assumption”** (biên quyết định nên nằm ở vùng mật độ dữ liệu thấp), do SVM có xu hướng đẩy biên ra xa các mẫu dữ liệu chưa gán nhãn.
--> Self-training tận dụng được cấu trúc phân bố của dữ liệu chưa gán nhãn để lan truyền tri thức --> mô hình trở nên mạnh mẽ và tổng quát hơn.

#### 5.1. Co-training:
Là phiên bản mở rộng của self-training, khi có **từ 2 classifiers trở lên** thực hiện quá trình phân loại và gán nhãn cho dữ liệu.

Và các máy này sẽ thường được huấn luyện cùng bộ dữ liệu - nhưng sử dụng các đặc trưng khác nhau (feature).
- Ví dụ: cùng là 2 máy phân loại các hình dạng (hình tứ giác và tam giác), nhưng một máy sẽ học dựa trên số cạnh của mẫu thử và máy kia sẽ học số đỉnh của mẫu thử.

Vậy sự khác biệt của các máy phân loại nhìn cùng 1 dataset thường được gọi là "sự khác biệt trong góc nhìn dữ liệu" (view)
Và đặc điểm của Co-training là bạn không bị giới hạn về chủng loại Classifier (bạn có thể kết hợp SVM và Logistic Regression hoặc SVM với mô hình CNN) để Co-training.

Yếu tố mà co-training quan tâm là tính chính xác của quá trình gán nhãn (Accuracy) và sự đồng thuận (Agreement) giữa đáp án (cách nhãn được gắn) cho các dữ liệu điểm từ các "**view**" khác nhau.




No tes:
mô hình AI phải có label --> ground_truth --> phạt khi loss cao --> knowlegde.

Mô hình deep learning gồm 2 phần 
- CNN (feature extraction)  --> biểu diễn các đặt trưng trong các vector 
- MLP (classifier) --> nhận vectors và xử lí đưa ra quyết định
--> related to classification
