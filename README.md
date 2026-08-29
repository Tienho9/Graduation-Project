# Graduation-Project: Path Planning Algorithms & Visualization

Dự án nghiên cứu, cải tiến và so sánh các thuật toán tìm đường (Path Planning) trên lưới bản đồ 2D, kết hợp giao diện đồ họa (GUI) và công cụ trực quan hóa kết quả.

---

## 📌 Tổng quan

Dự án tập trung triển khai và đánh giá hiệu năng giữa thuật toán tìm đường cổ điển và các biến thể kết hợp:
* **Traditional A\*** (`astar_trad.py`): Thuật toán A* truyền thống.
* **Improved A\*** (`astar_imp.py`): Thuật toán A* cải tiến tối ưu hóa hàm heuristic/chi phí.
* **Improved A\* with Greedy** (`astar_imp_with_greedy.py`): A* cải tiến kết hợp tìm kiếm tham lam (Greedy Search).
* **Improved A\* with ACO** (`astar_imp_with_aco.py`): A* cải tiến kết hợp thuật toán tối ưu hóa đàn kiến (Ant Colony Optimization - ACO).

---

## 📂 Cấu trúc thư mục

```text
├── astar_trad.py               # Triển khai thuật toán A* truyền thống
├── astar_imp.py                # Triển khai thuật toán A* cải tiến
├── astar_imp_with_greedy.py    # A* cải tiến kết hợp Greedy
├── astar_imp_with_aco.py       # A* cải tiến kết hợp ACO
├── gui.py                      # Giao diện người dùng đồ họa (GUI)
├── main.py                     # File thực thi chính
├── map_utils.py                # Các hàm tiện ích xử lý bản đồ
├── readmap.py                  # Đọc và phân tích dữ liệu ma trận bản đồ
├── visualization.py            # Trực quan hóa đường đi và kết quả so sánh
└── *.npy                       # Dữ liệu ma trận bản đồ mẫu (map2, chamgoc, new4, new7, new9, saved_map)
