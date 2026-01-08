# Tự Động Trích Xuất Thông tin Văn Bản và Số từ Hình Ảnh Biểu Đồ

**(Automated Textual and Numeric Information Extraction from Chart Images)**

Dự án này tập trung vào việc chuyển đổi dữ liệu phi cấu trúc từ hình ảnh biểu đồ cột (Bar Chart) thành dữ liệu có cấu trúc (bảng dữ liệu, file CSV/Excel). Hệ thống sử dụng phương pháp học sâu đa giai đoạn (Multi-stage Deep Learning) kết hợp giữa Thị giác máy tính (CV) và Xử lý ngôn ngữ tự nhiên (NLP).

## 📋 Mục tiêu chính

* 
**Trích xuất thành phần cơ sở:** Tự động phát hiện vùng văn bản và các thành phần đồ họa như cột, đường.


* 
**Phân loại vai trò ngữ nghĩa:** Xác định vai trò của văn bản (tiêu đề, nhãn trục, chú thích) bằng các mô hình Transformer tiên tiến.


* 
**Tái tạo dữ liệu:** Sử dụng thuật toán hình học để khôi phục lại bảng số liệu gốc từ hình ảnh với độ chính xác cao.



---

## 🏗 Kiến trúc hệ thống (Pipeline)

Quy trình xử lý được chia thành 5 giai đoạn chính:

1. **Text Detection & Recognition:** Sử dụng kiến trúc lai giữa **YOLO** (để phát hiện vùng chữ) và **PaddleOCR (PP-OCRv4)** để nhận dạng ký tự với độ chính xác cao.

2. **Text Role Classification:** Sử dụng mô hình đa phương thức **LayoutLMv3** để phân loại văn bản vào 9 vai trò khác nhau (Chart Title, Axis Title, Tick Label, v.v.) dựa trên nội dung, vị trí và hình ảnh.

3. **Axis Analysis:** Xác định hệ trục tọa độ và liên kết các nhãn trục (Tick Labels) với trục tương ứng để xây dựng thang đo pixel-to-value.

4. **Legend Analysis:** Sử dụng thuật toán **Hungarian** để ghép cặp chính xác giữa nhãn chú thích và ký hiệu màu sắc tương ứng.

5. **Data Extraction:** 
* Phát hiện các cột (Bar) bằng **YOLOv8s**.
* Sử dụng **ResNet50** để trích xuất đặc trưng hình ảnh và liên kết cột với chuỗi dữ liệu (Series).

---

## 🚀 Công nghệ sử dụng

* 
**Mô hình phát hiện:** YOLOv8 (phiên bản s và obb).
  
* 
**Nhận dạng văn bản:** PaddleOCR.

* 
**Hiểu tài liệu đa phương thức:** LayoutLMv3.

* 
**Trích xuất đặc trưng:** ResNet50.

* 
**Giao diện người dùng:** Web Interface (Hỗ trợ tải lên ảnh và xuất CSV).


---

## 📊 Kết quả thực nghiệm

Dự án được huấn luyện và đánh giá trên bộ dữ liệu **ICPR 2022 Chart-Info**.

| Tác vụ | Chỉ số đánh giá | Kết quả |
| --- | --- | --- |
| **Text Detection** | F1-Score | <br>**81.95%** |
| **Text Recognition** | Character Accuracy | <br>**92.11%** |
| **Role Classification** | Precision | <br>**98.90%** |
| **Plot Element Detection** | mAP@0.5 | <br>**97.40%** |

Công thức tính giá trị thực  của mỗi cột dựa trên thang đo được tính như sau:

Trong đó:

* 
: Giá trị tại đường cơ sở (trục hoành).


* 
: Chiều cao pixel của cột.


* 
: Tỉ lệ pixel-to-value được ước lượng.



---

## 👥 Thành viên thực hiện

* 
**Giảng viên hướng dẫn:** Mai Xuân Toàn, Trần Tuấn Anh, Huỳnh Văn Thống, Trần Hồng Tài.


* **Sinh viên thực hiện (Nhóm 9):**
* Lê Trần Tấn Phát (MSSV: 2312580).


* Bùi Ngọc Phúc (MSSV: 2312665).


* Nguyễn Hồ Quang Khải (MSSV: 2352538).


