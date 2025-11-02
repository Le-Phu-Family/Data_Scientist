Họ và tên: Lê Phú
Lớp: DH39KH02
MSSV: 030239230185

# **Ý TƯỞNG NGHIÊN CỨU KHOA HỌC LẦN 2**

---

## 1. **Tên đề tài**

**Tiếng Việt:**

> Mô hình phát hiện và giải thích gian lận giao dịch tài chính tại Việt Nam dựa trên đồ thị học bán giám sát và SHAP.

**Tiếng Anh:**

> An Explainable Semi-supervised Graph Attentive Network for Financial Fraud Detection in Vietnam.

---

## 2. **Tóm tắt nội dung nghiên cứu (Abstract)**

Sự gia tăng nhanh chóng của gian lận tài chính trong các hệ thống ngân hàng và fintech Việt Nam đòi hỏi những giải pháp trí tuệ nhân tạo (AI) tiên tiến có khả năng **phát hiện và giải thích** hành vi bất thường. Tuy nhiên, các phương pháp học có giám sát (supervised learning) truyền thống gặp khó khăn vì dữ liệu gian lận rất ít, không cân bằng, và thiếu nhãn.

Nghiên cứu này đề xuất một **mô hình học bán giám sát đồ thị có khả năng giải thích (Explainable Semi-supervised Graph Attentive Network – SGAT + SHAP)** nhằm phát hiện và lý giải các hành vi gian lận trong giao dịch tài chính tại Việt Nam.

Mô hình tận dụng mối quan hệ giữa các thực thể (người gửi, người nhận, thiết bị, merchant, ví điện tử) dưới dạng **đồ thị giao dịch**, đồng thời sử dụng **attention mechanism** để học trọng số quan trọng giữa các kết nối. Phần **Explainable AI (SHAP)** được tích hợp để diễn giải nguyên nhân quyết định, giúp mô hình minh bạch và đáng tin cậy hơn trong ứng dụng ngân hàng – fintech Việt Nam.

Kết quả kỳ vọng cho thấy mô hình có thể **giảm tỷ lệ cảnh báo sai (False Positives)**, duy trì **độ nhạy (Recall)** cao, và cung cấp **phân tích minh bạch về lý do dự đoán gian lận**, phù hợp với bối cảnh dữ liệu Việt Nam.

---

## 3. **Cơ sở lý thuyết và bối cảnh nghiên cứu**

Phát hiện gian lận (Fraud Detection) là một bài toán trọng yếu trong an ninh tài chính. Các mô hình truyền thống như Logistic Regression, Random Forest hay XGBoost yêu cầu nhiều dữ liệu có nhãn, trong khi ở Việt Nam, dữ liệu gian lận thường **ít và không cân bằng**.

Gần đây, các nghiên cứu tiên tiến như **Semi-supervised Graph Attentive Network (SGAT)** đã đạt kết quả tốt trong phát hiện gian lận vì khả năng **khai thác cấu trúc đồ thị** của mạng lưới giao dịch. Tuy nhiên, các mô hình này vẫn mang tính “hộp đen” (black-box), khó giải thích khi triển khai thực tế trong ngành ngân hàng, đặc biệt tại Việt Nam – nơi yêu cầu **tính minh bạch và kiểm toán AI** cao.

Do đó, việc kết hợp **SGAT với SHAP** mở ra hướng tiếp cận vừa hiệu quả, vừa có thể giải thích được các yếu tố ảnh hưởng tới quyết định gian lận, tăng niềm tin cho các tổ chức tài chính Việt Nam.

---

## 4. **Khoảng trống nghiên cứu (Research Gap)**

| Khoảng trống                                                                                                   | Giải thích                                                               |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| (1) Thiếu nghiên cứu ứng dụng mô hình **Graph-based Semi-supervised Learning** cho dữ liệu tài chính Việt Nam. | Hầu hết các mô hình được huấn luyện trên dữ liệu châu Âu hoặc Mỹ         |
| (2) Các mô hình đồ thị hiện nay **không có tính giải thích (non-transparent)**                                 | Ngân hàng không thể hiểu vì sao một giao dịch bị đánh dấu gian lận       |
| (3) Thiếu khung mô hình kết hợp **SGAT và Explainable AI (SHAP)**                                              | Đây là khoảng trống học thuật chưa được khai thác                        |
| (4) Dữ liệu Việt Nam đặc thù (nhiều giao dịch QR, ví điện tử, nhỏ lẻ) chưa được nghiên cứu                     | Cần một mô hình thích ứng với hành vi và cấu trúc giao dịch tại Việt Nam |

---

## 5. **Mục tiêu và câu hỏi nghiên cứu**

### 🎯 **Mục tiêu tổng quát**

Phát triển một mô hình học bán giám sát đồ thị có khả năng giải thích để phát hiện và giải thích hành vi gian lận tài chính tại Việt Nam.

### 🎯 **Mục tiêu cụ thể**

1. Biểu diễn dữ liệu giao dịch tài chính Việt Nam thành cấu trúc đồ thị (Graph-based Representation).
    
2. Xây dựng và huấn luyện mô hình **Semi-supervised Graph Attentive Network (SGAT)** cho phát hiện gian lận.
    
3. Tích hợp **SHAP** để giải thích quyết định của mô hình và xác định yếu tố ảnh hưởng lớn nhất đến dự đoán.
    
4. Đánh giá hiệu năng và tính minh bạch của mô hình so với các phương pháp truyền thống.
    

---

### ❓ **Câu hỏi nghiên cứu**

| Mã  | Câu hỏi nghiên cứu                                                                               | Mục đích                  |
| --- | ------------------------------------------------------------------------------------------------ | ------------------------- |
| RQ1 | Làm thế nào để biểu diễn giao dịch tài chính Việt Nam thành đồ thị phù hợp cho học bán giám sát? | Thiết kế cấu trúc đồ thị  |
| RQ2 | Mô hình SGAT có cải thiện khả năng phát hiện gian lận so với các mô hình truyền thống không?     | Đánh giá hiệu năng        |
| RQ3 | SHAP có thể giúp giải thích các yếu tố ảnh hưởng đến quyết định gian lận như thế nào?            | Tăng tính minh bạch       |
| RQ4 | Mức độ khả thi và hiệu quả của mô hình SGAT + SHAP trong bối cảnh dữ liệu Việt Nam ra sao?       | Đánh giá ứng dụng thực tế |

---

## 6. **Phương pháp nghiên cứu (Methodology)**

### (1) **Thu thập và xử lý dữ liệu**

- Nguồn dữ liệu:
        
    - dữ liệu ẩn danh từ đối tác ngân hàng / fintech (cần tìm).
        
- Xử lý:
    
    - Làm sạch, ẩn danh, chuẩn hóa (VND, thời gian, kênh giao dịch).
        
    - Trích xuất đặc trưng: số tiền, thời gian, merchant, thiết bị, vị trí, loại giao dịch.
        

### (2) **Xây dựng đồ thị giao dịch**

- **Nodes:** tài khoản, thiết bị, merchant, ví điện tử.
    
- **Edges:** giao dịch, tương tác giữa người dùng – merchant.
    
- **Features:** hành vi, tần suất, giá trị, địa lý, loại thiết bị.
    

### (3) **Huấn luyện mô hình SGAT (Semi-supervised Graph Attentive Network)**

- Áp dụng cơ chế attention để học trọng số các kết nối.
    
- Loss function gồm:
    
    - Cross-entropy cho dữ liệu có nhãn.
        
    - Consistency loss cho dữ liệu không nhãn.
        
- Framework: PyTorch Geometric hoặc DGL.
    

### (4) **Giải thích bằng SHAP**

- Sử dụng **GraphSHAP hoặc DeepSHAP** để tính giá trị ảnh hưởng của các feature hoặc edge.
    
- Trực quan hóa qua **heatmap hoặc edge importance graph**.
    
- Phân tích định tính: giải thích tại sao mô hình dự đoán một giao dịch là gian lận.
    

### (5) **Đánh giá hiệu năng**

|Nhóm chỉ số|Metric|Mục tiêu|
|---|---|---|
|Phát hiện gian lận|Precision, Recall, F1, AUC|Đánh giá chính xác mô hình|
|Cảnh báo sai|False Positive Rate|Giảm cảnh báo sai|
|Giải thích|Mean SHAP Value, Feature Consistency|Đánh giá tính minh bạch|
|Ứng dụng thực tế|Time cost, interpretability|Kiểm tra khả năng triển khai|

---

## 7. **Kết quả kỳ vọng**

- Mô hình **SGAT + SHAP** có thể phát hiện và **giải thích** gian lận giao dịch hiệu quả.
    
- Giảm **False Positive Rate**, duy trì **Recall cao**, đồng thời **tăng tính minh bạch AI**.
    
- Đề xuất **framework ứng dụng cho ngân hàng/fintech Việt Nam**.
    
- Đóng góp học thuật: mở rộng Explainable AI cho Graph-based Fraud Detection.
    

---

## 8. **Đóng góp khoa học và thực tiễn**

|Loại đóng góp|Nội dung|
|---|---|
|**Học thuật**|Đề xuất mô hình SGAT + SHAP đầu tiên cho phát hiện gian lận tại Việt Nam|
|**Kỹ thuật**|Framework đồ thị bán giám sát có khả năng giải thích|
|**Thực tiễn**|Hỗ trợ ngân hàng Việt Nam triển khai AI minh bạch và đáng tin cậy|
|**Xã hội**|Góp phần phòng chống gian lận tài chính và nâng cao an toàn thanh toán số|

---

## 9. **Tài liệu tham khảo chính**

1. 1. Khalid, A. R., Owoh, N., Uthmani, O., Ashawa, M., Osamor, J., & Adejoh, J. (2024). Enhancing Credit Card Fraud Detection: An Ensemble Machine Learning Approach. Big Data and Cognitive Computing, 8(1), 6. https://doi.org/10.3390/bdcc8010006  
2. Hafez, I.Y., Hafez, A.Y., Saleh, A. et al. A systematic review of AI-enhanced techniques in credit card fraud detection. J Big Data 12, 6 (2025). https://doi.org/10.1186/s40537-024-01048-8  
3. van Engelen, J.E., Hoos, H.H. A survey on semi-supervised learning. Mach Learn 109, 373–440 (2020).  https://doi.org/10.1007/s10994-019-05855-6  
4. Teksands . (2021, October 26). Introduction to Semi-Supervised Learning | TeksandsAI. Teksands. https://teksands.ai/blog/semi-supervised-learning 
5. Almuteer, A. H., Aloufi, A. A., Alrashidi, W. O., Alshobaili, J. F., & Ibrahim, D. M. (2021). Detecting Credit Card Fraud using Machine Learning. International Journal of Interactive Mobile Technologies (iJIM), 15(24), pp. 108–122. https://doi.org/10.3991/ijim.v15i24.27355 
6. Faruk, Nayab & Tariq, Ahmad & Oladele, Sunday & Gok, Mooale. (2025). Explainable AI (XAI) for Fraud Detection: Building Trust and Transparency in AI-Driven Financial Security Systems. https://www.researchgate.net/publication/390235753_Explainable_AI_XAI_for_Fraud_Detection_Building_Trust_and_Transparency_in_AI-Driven_Financial_Security_Systems
7. Hosseini Chagahi, M., Delfan, N., Mohammadi Dashtaki, S., Moshiri, B., & Piran, M. J. (2024). _An innovative attention-based ensemble system for credit card fraud detection_. arXiv preprint arXiv:2410.09069. https://arxiv.org/abs/2410.09069 
8. Faruk, N., Tariq, A., Oladele, S., & Gok, M. (2025). _Explainable AI (XAI) for fraud detection: Building trust and transparency in AI-driven financial security systems_. Retrieved from https://www.researchgate.net/publication/390235753_Explainable_AI_XAI_for_Fraud_Detection_Building_Trust_and_Transparency_in_AI-Driven_Financial_Security_Systems

