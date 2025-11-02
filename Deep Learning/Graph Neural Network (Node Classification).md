tìm hiểu neural network và lí thuyết đồ thị (graph theory) trước

1. What is Graph Data Around Us?
![[Pasted image 20251023165801.png]]
Nhìn hình ở trên, ta có:
điểm chung của chúng là đều có mối quan hệ (relationship) giữa các nodes bằng các cạnh (edges)
- Graph Classification: là ta đã có sẵn bộ dữ liệu chẳng hạn và ta đưa vào nguyên mô hình graph này (graph đã hoàn chỉnh) và chúng sẽ phân loại ra T/F or Y/N.
	input: feature --> output: class
- Node Classification: bài toán này khác với loại ở trên thay vì dùng toàn bộ graph và bộ dữ liệu có sẵn để dự đoán lớp. Chúng ta sẽ có những node đã được label sẵn và nhiệm vụ của chúng ta là dự đoán hay label các node unlabel. (graph chưa được hoàn chỉnh).
- Link Prediction: ví dụ ta có thể nhìn hình rằng các node là các video. Và làm sao để khi mình coi vid này xog thì nó sẽ đề xuất vid khác cho mình. (recommendation systems). Chẳng hạn như tập 1 và tập 2 của một bộ phim phải có sự liên kết giữa 2 tập này. Tức là khi coi xong tập 1 phải đề xuất tập 2 chứ ko được tập 2,3,4 của bộ phim khác chẳng hạn.
- Community Detection: tức là đi tìm các sample nào có đặc trưng (feature) giống nhau thì đưa vào một nhóm.
- Graph Embedding: tức là từ một graph rồi mình đi embedding chúng để thành một bộ số của một feature rồi có thể đem vào một mô hình ML khác để xử lí các bước tiếp theo.
- Graph Generation: 

1.1. Graph Definition:
A graph represents the relations (edges) between a collection of entities (nodes).
- Vertex (nodes): có thể được coi là các sample trong mẫu dữ liệu
- Edge (link): được coi là đường thể hiện mối quan hệ giữa các node
- Global (master node) embedding: đó là embedding của node thể hiện được các mối quan hệ của các node 
![[Pasted image 20251023172944.png]]

1.2. Graphs are everywhere
- để đưa được dữ liệu vào các node của neural network, chúng ta cần có feature embedding của chúng.
- Vậy khi xây dựng một graph neural network, chúng ta cần:
	- Embedding của các node.
	- Edges Embedding của nodes
	--> bằng adjency matrix 
![[Pasted image 20251023174314.png]]



2. Understand Graph Neural Network
- the goal of GNN is to transform node features to features that are aware of the graph structure
	- tức là làm sao chuyển đổi từ các node embedding bằng cách sử dụng một cái transformer function để thành cái node embedding (hidden layer) thể hiện được các mối quan hệ giữa các node








3. Understand Graph Convolutional Neural Network 
4. Node Classification with Cora Citation Dataset
5. 