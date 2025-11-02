## 1.  Anti-Money Laundering (AML) 
- Tình trạng:
	- LaundroGraph method using SS graph representation learning
	- **Graph Contrastive Pre-training for AML (GCPAL) - sử dụng học đối chiếu đồ thị (graph contrastive learning)**
		- **Đã mở rộng khi có supervised classification in pre-training process and add graph augmentations**** 
	-  Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering Detection in Bitcoin
		- Trong môi trường khác (tiền mã hóa) --> self-supervised + đồ thị có ứng dụng tốt trong các domain AML khác nhau.
- ### Motivations:
	- Embedding in real-time, temporal (ví dụ bằng cách xây dựng các snapshot đồ thị qua thời gian hoặc dùng mô hình kế tiếp để theo dõi sự thay đổi hành vi của khách hàng theo thời gian)
	- XAI (explainability) cho các cảnh báo bất thường và tuân thủ quy định
	- Dùng negative sampling để lấy mẫu 'khó' --> giúp mô hình học tốt hơn
	- Ứng dụng vào domain khác như risk scoring in E-commerce/
- # After research:
	- GCPAL + Federated Learning : Anti-Money Laundering + Cyphersecurity in Banking and cryptocurrency domain.
		🔹 **Lấy mẫu đồ thị con (Subgraph Sampling):** Huấn luyện trên các phần đồ thị đại diện để giảm chi phí tính toán nhưng vẫn giữ thông tin cấu trúc.  
		🔹 **Kỹ thuật cô đọng đồ thị (Graph Condensation):** Dùng các phương pháp như rút gọn cạnh (sparsification) hoặc giảm chiều để giảm bộ nhớ mà không làm mất quan hệ quan trọng.
	- Ứng dụng trên domain khác:
		- detect bảo hiểm
		- Rumor Detection 



**QUY TRÌNH TÌM ĐỀ TÀI NGHIÊN CỨU**
A Hybrid Semi-supervised Learning Framework for Credit Card Fraud Detection: Integrating Autoencoders and Ensemble Classifiers
1. Identify areas of interest
- Data Science/AI (Hybrid Semi-supervised Learning)
- CCFD (Credit Card Fraud Detection)
- Finance and Banking Industry

1. Finding research gap
- AI-Enhanced Credit Card Fraud Detection: A Systematic Review (local) --> semi-supervised learning/active learning to solve imbalanced/biased problems in dataset in Finance and Banking Industry

3. Identify the goals and the approach of research:
- Nghiên cứu này áp dụng cách tiếp cận học bán giám sát kết hợp lai (**hybrid semi-supervised approaches**) tích hợp kỹ thuật phát hiện vật thể ngoại lai (outlier detection) không giám sát trong Credit Card Fraud Detection.
- Học giám sát: Ensemble Learning (SVM, KNN, RF, Boosting)
- Học không giám sát + thuật toán phát hiện vật thể ngoại lai: AEs(DL)/**Autoencoders**(AE)/ **K-means Clustering** (hoặc LOF)
--> giải quyết vấn đề mất cân bằng/thiên lệch về dữ liệu.
==> Mục tiêu chính của tiếp cận lai này là tận dụng **dữ liệu chưa được gán nhãn khổng lồ** (chủ yếu là giao dịch hợp lệ) để giúp mô hình hiểu rõ hơn về **mẫu hành vi bình thường**, từ đó tăng cường khả năng phát hiện các trường hợp gian lận hiếm gặp (lớp thiểu số).

4. Identify the goals and the approach of research:
RQ1: Làm thế nào 
RQ2: Liệu việc áp dụng cách tiếp cận học bán giám sát lai có giúp cải thiện được vấn đề mất cân bằng/thiên lệch dữ liệu?
RQ3: Liệu mô hình có thể phát hiện và học được các trường hợp gian lận hiếm gặp không mà không làm tăng False Positive không? 
RQ4: Liệu mô hình có thể
RQ5:

5. Identify methodology:


Bước 1: Phân tách và Chuẩn bị Dữ liệu

1. **Phân loại Dữ liệu:**

    ◦ **Tập Dữ liệu Đã gán nhãn (****D_L****):** Bao gồm một lượng nhỏ giao dịch đã xác nhận là **Gian lận** (lớp thiểu số) và **Hợp lệ** (lớp đa số).

    ◦ **Tập Dữ liệu Chưa gán nhãn (****D_{UL}****):** Bao gồm phần lớn các giao dịch chưa được xác minh nhãn, được giả định chủ yếu là hợp lệ (bình thường). Mục tiêu là sử dụng dữ liệu này để cải thiện tính mạnh mẽ của mô hình.

2. **Xử lý Dữ liệu Lớn (Big Data):** Vì dữ liệu CCFD thường là tập dữ liệu lớn với khối lượng giao dịch khổng lồ, cần sử dụng các khung xử lý dữ liệu phân tán như **Apache Spark** hoặc **MapReduce** để cho phép xử lý song song. Điều này giúp mô hình giải quyết được vấn đề **khả năng mở rộng (scalability)** và **nhu cầu tính toán cao**.

Bước 2: Kỹ thuật Học Không giám sát (Unsupervised Component) – Phát hiện Vật thể Ngoại lai

Đây là thành phần "không giám sát" của mô hình lai, chuyên về việc học cấu trúc bình thường của dữ liệu.

1. **Đào tạo Mô hình Phát hiện Ngoại lai:**

    ◦ Sử dụng các kỹ thuật Học không giám sát, chẳng hạn như **Autoencoders (AEs)** hoặc các mô hình phát hiện vật thể ngoại lai khác, được đào tạo chủ yếu trên **Tập** **D_{UL}** (dữ liệu chưa được gán nhãn).

    ◦ Mục đích là để mô hình học được **đặc trưng và phân phối** của các giao dịch hợp lệ, bình thường.

2. **Đánh dấu Vật thể Ngoại lai:**

    ◦ Mô hình AE (hoặc mô hình phát hiện ngoại lai tương tự) sẽ đo lường **sai số tái tạo (reconstruction error)** cho mỗi giao dịch trong DUL​.

    ◦ Những giao dịch có sai số tái tạo cao (tức là chúng khác biệt đáng kể so với hành vi bình thường) sẽ được đánh dấu là **vật thể ngoại lai (outliers)** hoặc **gian lận tiềm năng**.

Bước 3: Kỹ thuật Học Có giám sát (Supervised Component) – Phân loại và Tinh chỉnh

Đây là thành phần "có giám sát" của mô hình lai, chuyên sử dụng các nhãn đáng tin cậy để tinh chỉnh dự đoán.

1. **Đào tạo Mô hình Cơ sở:**

    ◦ Đào tạo một mô hình phân loại có giám sát (ví dụ: **LSTM**, **CNN**, **Random Forest (RF)**, hoặc **XGBoost**) trên **Tập** **D_L** (dữ liệu đã được gán nhãn ban đầu).

    ◦ Bước này đảm bảo mô hình có thể nhận diện các mẫu gian lận đã được xác nhận.

Bước 4: Tích hợp Lai (Hybrid Integration) và Tự đào tạo (Self-Training)

Đây là bước kết hợp hai thành phần để giải quyết sự mất cân bằng lớp.

1. **Gán Nhãn Giả (Pseudo-Labeling):**

    ◦ Sử dụng mô hình có giám sát đã được đào tạo trong Bước 3 để dự đoán nhãn cho các giao dịch trong **Tập** **D_{UL}** đã được đánh dấu là **vật thể ngoại lai/gian lận tiềm năng** ở Bước 2.

    ◦ Chỉ chọn các giao dịch ngoại lai mà mô hình có giám sát có độ tin cậy dự đoán cao (ví dụ: xác suất > 95% là gian lận) và gán cho chúng **nhãn giả (pseudo-labels)**.

2. **Mở rộng Tập Huấn luyện:**

    ◦ Thêm các giao dịch ngoại lai có nhãn giả này vào tập huấn luyện ban đầu (DL​).

3. **Đào tạo lại Mô hình Lai (Iterative Retraining):**

    ◦ Đào tạo lại mô hình có giám sát (hoặc mô hình tổng hợp/lai cuối cùng) trên tập dữ liệu mở rộng này.

    ◦ Quá trình này được lặp lại, cho phép mô hình sử dụng hiệu quả **dữ liệu chưa được gán nhãn**, giúp mô hình học được các mẫu phức tạp hơn và cải thiện khả năng tổng quát hóa.

Bước 5: Tối ưu hóa Mô hình và Khắc phục Hạn chế

Để đảm bảo hiệu quả của mô hình lai phức tạp:

1. **Tối ưu hóa Siêu tham số (Hyperparameter Optimization - HPO):**

    ◦ Sử dụng các **Thuật toán Tối ưu hóa Meta-heuristic (MHO)** như **Thuật toán Di truyền (GA)** để tinh chỉnh các tham số phức tạp của mô hình lai. HPO là cần thiết vì các mô hình DL/ML phức tạp thường nhạy cảm với việc điều chỉnh tham số.

2. **Sử dụng Mô hình Tổng hợp (Ensemble Models):**

    ◦ Phát triển các mô hình lai bằng cách tích hợp nhiều mô hình (ví dụ: kết hợp đầu ra của CNN, LSTM và RF) để tăng cường độ mạnh mẽ và độ chính xác của việc phát hiện. Việc này giúp giảm thiểu **số lượng cảnh báo sai (false positives)**.
    
3. **Học nhạy cảm với chi phí (Cost-sensitive Learning):**
    ◦ Triển khai các phương pháp học nhạy cảm với chi phí để tối ưu hóa việc phát hiện mà không bị **quá khớp (overfitting)** với các lớp đa số. Điều này rất quan trọng để quản lý sự mất cân bằng lớp nghiêm trọng (ví dụ: 0.17% giao dịch gian lận)

**6. Phương pháp nghiên cứu tạm thời**

Nghiên cứu gồm 5 giai đoạn chính:

**Giai đoạn 1: Chuẩn bị và tiền xử lý dữ liệu**

- **Phân chia dữ liệu:**

- _D_L_ (2%) – Dữ liệu có nhãn (fraud / non-fraud).
- _D_UL_ (98%) – Dữ liệu chưa có nhãn.

- **Tiền xử lý:** Chuẩn hóa đặc trưng (MinMaxScaler), loại bỏ trùng lặp, chuẩn hóa giá trị.
- **Công cụ:** Python, TensorFlow, Apache Spark (xử lý dữ liệu lớn).

**Giai đoạn 2: Thành phần học không giám sát – Autoencoder**

- Huấn luyện Autoencoder trên D_UL để học mẫu hành vi giao dịch bình thường.
- Tính toán lỗi tái tạo (reconstruction error).
- Giao dịch có lỗi cao được xem là **ngoại lai (outlier)** – nghi ngờ gian lận.

**Giai đoạn 3: Thành phần học có giám sát – Mô hình tập hợp**

- Huấn luyện mô hình Random Forest, XGBoost, và LightGBM trên D_L.
- Đánh giá hiệu năng ban đầu theo AUC, Precision, Recall, F1-score.

**Giai đoạn 4: Kết hợp lai và cơ chế tự huấn luyện**

- Kết hợp kết quả từ AE và mô hình có giám sát:

- Sử dụng kết quả AE để chọn các mẫu nghi ngờ.
- Dựa vào xác suất dự đoán cao (>0.95) từ mô hình có giám sát để **gán nhãn giả (pseudo-label)**.
- Thêm các mẫu này vào tập D_L và huấn luyện lại.
- Lặp lại cho đến khi hiệu năng ổn định.

**Giai đoạn 5: Tối ưu và đánh giá**

- **Tối ưu siêu tham số:** Bằng thuật toán di truyền (Genetic Algorithm).
- **Điều chỉnh chi phí sai số (Cost-sensitive Learning)** để xử lý mất cân bằng.
- **Đánh giá mô hình theo các chỉ số:**

- AUC-ROC, Precision, Recall, F1-score, False Positive Rate.

- **So sánh với các mô hình cơ sở:**

- Supervised (RF, XGBoost)
- Autoencoder (unsupervised)
- Semi-supervised cơ bản (Label Propagation, PU Learning)
- Mô hình lai đề xuất.





# **Ý tưởng lần 2**:
> Vì tôi đã tìm ra **A Semi-supervised Graph Attentive Network** - một cách tiếp cận kiểu Hybrid Semi-supervised Learning For Fraud Detection (cùng chủ đề) - đặc biệt là đã đặt được **trạng thái SOTA** ở hiện tại.
> Thế nên, thay vì tạo ra một thuật toán/approach mới hoàn toàn dựa trên research gap thì mình sẽ chuyển sang ứng dụng thực tiễn mô hình vừa thể hiện mối quan hệ và có thể giải thích được ở Việt Nam bằng Semi-supervised Graph Attentive Network và SHAP.

**Mục đích:**
- Giải thích các đặc trưng đã ảnh hưởng lớn khi ra quyết định --> đảm bảo tính minh bạch và tính pháp lý đối với thị trường ở Việt Nam
- Giải quyết các bài toán mất cân bằng/thiên lệch về dữ liệu (đặc biệt là trong Fraud Detection)
- Tận dụng được dữ liệu không nhãn lớn để hỗ trợ cho classifier học được các insights tiềm ẩn.
- Khả năng suy diễn mạnh (inductive learning), đa khía cạnh của classfiers (semiGAN) nhằm giải thích mối quan hệ phức tạp giữa các nodes bằng link trong mạng lưới đồ thị.

