- Tập trung định hướng tương lai của doanh nghiệp, ví dụ như sản phẩm mới triển khai trong vòng 6 tháng tới sẽ như thế nào hay nên đầu tư vào bao nhiêu tiền
- thường phải sử dụng các dữ liệu thô và cả dữ liệu đã được xử lí 
- Xác suất thống kê: A/B testing và Hypothesis testing (task cơ bản)
- Giải tích, đại số tuyến tính, bài toán tối ưu
- Machine learning: (recommendation systems, predictive model)
- Deep Learning & Neural Networks
- Thành thạo: SQL, Python
- Software Development: Git, Docker, CI/CD, DevOps.
--> ưu tiên accuracy, making data-driven decisions
--> tập trung vào tìm hiểu, nghiên cứu làm sao ra được 1 framework tốt để giải quyết bài toán một cách hiệu quả.
như một nhà khoa học trong phòng lab, phải thử nghiệm để tạo ra nhiều cái tiên nghiệm khác nhau rồi tìm ra sản phẩm tốt nhất. Nhưng chỉ gói gọn trong phòng lab. Vậy nên MLE sẽ đem nó đi deploy rộng rãi hơn. 
Machine Learning Engineering
- optimize inference bằng (ONNX, TensorRT)
	- ONNX: định dạng trung gian, ví dụ convert từ Keras sang Pytorch cần có định dạng trung gian.
	- TensorRT: đó là thư viên đặc biệt của NVIDIA, tối ưu runtime (nhưng chỉ tối ưu trên card đồ họa NVIDIA)
	--> trình tự: convert to ONNX rồi đến TensorRT rồi tích hợp vào các thiết bị đi deploy.
- Làm sao để deploy models vào hệ thống thực có thể tối ưu hơn về mặt GPU, running time, ... nhằm scalability, latency
--> Convert models framework vào hệ thống thực.


Cuda: kiến trúc lập trình song song dành cho các GPU NVIDIA. hiểu được cách cài đặt, sử dụng sao cho tối ưu.
**Học thêm về keras, pytorch framework.**
**Ôn lại toán, docker**
**Học thêm về attention, transformer và deep learning**
