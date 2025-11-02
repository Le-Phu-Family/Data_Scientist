![[Pasted image 20251009023010.png]]
Attention - đứng sau và là nền tảng cho kiến trúc Transformer đã làm bùng nổ và thay đổi cả thế giới AI. 🤖✨ Thứ đứng sau sự thành công của các mô hình ngôn ngữ hiện tại như GPT, BERT, ...
## I. Tại sao Attention lại xuất hiện và vấn đề "Nút thắt cổ chai"?

- Vì nó đã giúp giải được bài toán "Nút thắt cổ chai" mà các mô hình deep learning trước 2017 như RNN - mô hình mạng neuron hồi quy đang mắc phải. 
![[Pasted image 20251009030236.png]]
```
Ví dụ: bạn có input đầu vào là một câu dài ơi là dài. Và bạn muốn phân tích context hay semantic (ngữ nghĩa) của chúng. 
Và model đồ của bạn phải chạy một cách tuần tự rất nghiệm ngặt và các thông tin của bạn bị nén vào một hộp thông tin (vector) với kích thước cố định (fixed size)
```
- Vì chúng được học một cách **tuần tự** là đi qua từng từ trong câu một cách rất nghiêm ngặt và giới hạn về trí nhớ - tức là khi học (chạy tuần tự) thì khi chạy dần về cuối câu, nó đã quên ý nghĩa của đoạn đầu mất rồi :(( (làm sao hiểu được toàn bộ ngữ nghĩa trong câu đây!!??) -> Vấn đề đó đã tạo ra "nút thắt thông tin".
--> tốn thời gian và trí nhớ kém (bài toán cần giải).

Vậy mục đích mà Attention xuất hiện là để có thể làm việc các bài toán dịch thuật ngôn ngữ (Machine Translation) - và đầu vào chính là các từ - các câu (Sequential Datum) - nhằm nhanh hơn và có thể hiểu được hết ngữ nghĩa trong câu. ==hiệu quả hơn khi hạn chế (tối ưu hóa) sai số ==
```
- Như trong một câu “I am kicking the ball”
- Thì nếu cho RNN dịch câu này thì RNN sẽ tra từng từ một như I, am, kicking, the, ball
- Trong khi nếu muốn dịch nhanh thì mô hình chỉ cần “Skimming” và “Scanning” các chi tiết quan trong đến bối cảnh như “I”, “kicking” , và “ball” : mang toàn bộ ngữ cảnh của câu - hạn chế chú ý hơn vào các chi tiết chỉ mang tính chất “cú pháp”
- Đây có thể là minh hoạ với tình huống “nhiều khi bạn nói sai ngữ pháp, nhưng người đối diện vẫn có thể hiểu được bạn đang nói những gì” - thì RNN không thể có khả năng đó được !
```

### Một số thuật ngữ cơ bản!!!

#### 1.1.  **Mạng neuron (neural networks)**
Mạng neuron (neural networks) là một kiểu mô hình ML cơ bản và được lấy ý tưởng từ cấu trúc neuron thần kinh của não bộ con người với các thành phần sau: --> deep learning  

-  Input (đầu vào) của bạn: (có thể là 1 tấm ảnh, một câu, ….), nó sẽ được mã hóa thành bộ các con số (gọi là vector A), mỗi các số trong vector A được coi là một node hoặc một neuron.
--- để mã hóa một câu thì chúng ta dùng embedding (nhúng) để đưa các dữ liệu rời rạc (words) thành các dãy số (vector) -> để máy tính có thể học và hiểu được mối liên hệ của chúng --> store chúng vào matrices hay vector databases để tiện cho mô hình xử lí.

-  Các neuron (các con số trong A: tạm gọi là x) sẽ được kết nối với nhau bằng connection (links) và mỗi connection này sẽ có một giá trị thể hiện mức độ “nặng” của từng liên kết trong toàn bộ giá trị đầu vào (độ nặng này gọi là Weight: W)

-  Để neuron có thể kết nối với nhau thì chỉ cần lấy:
	o   Đầu vào * trọng số (weight) à kết quả của 1 neuron
	o   Sau đó, lặp lại các bước trên với toàn bộ các giá trị inputs của bạn (gọi là tính tổng) + một giá trị bias à kết quả Z

-        Nhưng kết quả Z này chưa kết quả mà máy cần học, nhưng chưa đủ để máy bạn sử dụng được (vì kết quả ấy chỉ toàn là số 😊) à nên nó cần được kích hoạt thông qua một hàm kích hoạt (Activation Function) --> đầu ra của hàm chính là đầu ra mà mô hình trả lại - Y. 
![[Pasted image 20251009023227.png]] 
#### 1.2. Sequential Data (dữ liệu dạng chuỗi) 
là kiểu dữ liệu khi mà thứ tự của các thành phần tạo nên một dữ liệu điểm (data poinits) -> lại ảnh hưởng đến giá trị của điểm đó.
```
Ví dụ: trong một câu thứ tự của các từ trong câu (thành phần cấu thành dữ liệu điểm) ảnh hưởng đến nghĩa của toàn bộ câu văn trên (giá trị của dữ liệu). Vì vậy câu hay textual data được coi là Sequential Data
Còn nhiều ví dụ khác về Sequential Data: bao gồm âm thành, sóng, etc.
```
![[Pasted image 20251009025455.png]]
Chú ý: nó khác Embedding nha! Embedding là kỹ thuật mã hóa -  biến từng chữ (dữ liệu rời rạc) thành các dãy số (vector) trong không gian chiều dữ liệu cao. Dãy vector: `[[0.8, 0.1, ...], [0.7, 0.3, ...], ...]`
Trong khi, Sequential data là dữ liệu đầu vào (input) có thứ tự - nghĩa là chuỗi phần tử **có ý nghĩa.** Ví dụ: `["Tôi", "yêu", "phở", "Việt Nam"]`

## II. Giải thích về Attention và bàn về Transformer?













## III. Cơ chế tập trung (Attention) và ưu điểm của nó?