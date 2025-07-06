import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
                            QGroupBox, QGridLayout, QCheckBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from main import run_algorithms, calculate_overall_scores
from map_utils import InteractiveMapEditor


class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)

    def back(self, *args, **kwargs):
        super().back(*args, **kwargs)
        self.canvas.draw_idle()

class PathfindingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giao Diện Thuật Toán Tìm Đường")
        self.setGeometry(100, 100, 1200, 800)
        
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5) 
        
        
        map_group = QGroupBox("Thiết Lập Bản Đồ")
        map_layout = QVBoxLayout()
        
        
        size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 100)
        self.width_spin.setValue(20)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 100)
        self.height_spin.setValue(20)
        size_layout.addWidget(QLabel("Chiều Rộng:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Chiều Cao:"))
        size_layout.addWidget(self.height_spin)

        
        self.targets_spin = QSpinBox() 
        self.targets_spin.setRange(0, 10)
        self.targets_spin.setValue(1)
        size_layout.addSpacing(20) 
        size_layout.addWidget(QLabel("Số Lượng Mục Tiêu:"))
        size_layout.addWidget(self.targets_spin)
        size_layout.addStretch() 

        map_layout.addLayout(size_layout) 
        
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "Đồng Nhất",
            "Kho Hàng",
            "Thành Phố"
        ])
        
        
        self.env_combo.currentTextChanged.connect(self.on_environment_type_changed)
        
        
        self.obstacle_spin = QDoubleSpinBox()
        self.obstacle_spin.setRange(0, 0.4)
        self.obstacle_spin.setValue(0.2)
        self.obstacle_spin.setSingleStep(0.1)
        
        env_obstacle_layout = QHBoxLayout()
        env_obstacle_layout.addWidget(QLabel("Loại Môi Trường:"))
        env_obstacle_layout.addWidget(self.env_combo, stretch=1) 
        env_obstacle_layout.addSpacing(20) 
        self.obstacle_label = QLabel("Tỷ Lệ Chướng Ngại Vật:")
        env_obstacle_layout.addWidget(self.obstacle_label)
        env_obstacle_layout.addWidget(self.obstacle_spin)
        env_obstacle_layout.addStretch() 

        map_layout.addLayout(env_obstacle_layout) 
        
        button_layout = QHBoxLayout()
        self.create_btn = QPushButton("Tạo Bản Đồ Mới")
        self.load_btn = QPushButton("Tải Bản Đồ")
        self.save_btn = QPushButton("Lưu Bản Đồ")
        button_layout.addWidget(self.create_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        map_layout.addLayout(button_layout)

        edit_layout = QHBoxLayout()
        self.edit_btn = QPushButton("Tạo Bản Đồ Thủ Công")
        self.duplicate_btn = QPushButton("Tạo bản đồ tương tự")
        edit_layout.addWidget(self.edit_btn)
        edit_layout.addWidget(self.duplicate_btn)
        map_layout.addLayout(edit_layout)
        
        map_group.setLayout(map_layout)
        left_layout.addWidget(map_group)
        
        algo_group = QGroupBox("Thiết Lập Thuật Toán")
        algo_layout = QVBoxLayout()
        algo_layout.setSpacing(0)
        algo_layout.setContentsMargins(5, 0, 5, 0) 
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "Tất Cả Thuật Toán",
            "A*",
            "ImpA*",
            "ImpA*G",
            "ImpA*ACO"
        ])

        algo_select_layout = QHBoxLayout()
        algo_select_layout.addWidget(QLabel("Chọn Thuật Toán:"))
        algo_select_layout.addWidget(self.algo_combo, stretch=1) 
        algo_select_layout.addStretch() 

        algo_layout.addLayout(algo_select_layout)
        
        self.run_btn = QPushButton("Chạy Thuật Toán")
        algo_layout.addWidget(self.run_btn)
        
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)
        
        visibility_group = QGroupBox("Hiển Thị Đường Đi")
        visibility_layout = QVBoxLayout()

        self.path_checkboxes = {}
        self.path_lines = {}

        self.algorithm_names = ["A*", "ImpA*", "ImpA*G", "ImpA*ACO"]

        for name in self.algorithm_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True) 
            checkbox.toggled.connect(lambda checked, algo=name: self.toggle_path_visibility(algo, checked))
            visibility_layout.addWidget(checkbox)
            self.path_checkboxes[name] = checkbox

        visibility_group.setLayout(visibility_layout)
        left_layout.addWidget(visibility_group)

        self.order_label = QLabel("")
        self.order_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.order_label.setStyleSheet("color: red;")
        self.order_label.setWordWrap(True)
        left_layout.addWidget(self.order_label)

        layout.addWidget(left_panel, stretch=1)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111) 
        
        self.toolbar = CustomToolbar(self.canvas, right_panel)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1) 

        compare_group = QGroupBox("So Sánh Thuật Toán")
        compare_layout = QVBoxLayout()

        self.comparison_table = QTableWidget()

        compare_layout.addWidget(self.comparison_table)
        compare_group.setLayout(compare_layout)
        right_layout.addWidget(compare_group, stretch=0)

        layout.addWidget(right_panel, stretch=5)
        
        self.create_btn.clicked.connect(self.create_new_map)
        self.load_btn.clicked.connect(self.load_map)
        self.save_btn.clicked.connect(self.save_map)
        self.run_btn.clicked.connect(self.run_algorithm)
        self.edit_btn.clicked.connect(self.edit_map_manually)  
        self.duplicate_btn.clicked.connect(self.duplicate_map)

        self.current_map_file = None
        self.map_editor = None
        self.paths = None
        self.visited_nodes_all_algos = None
        self.orders = None

        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.installEventFilter(self)

        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(False)

    def on_environment_type_changed(self, env_type):
        if env_type.lower() in ["kho hàng", "thành phố"]:
            self.obstacle_spin.hide()
            self.obstacle_label.hide()
        else:
            self.obstacle_spin.show()
            self.obstacle_label.show()
        
    def on_map_click(self, event):
        """Handle mouse clicks on the Matplotlib canvas for map editing."""
        if not self.map_editor or event.inaxes != self.figure.gca():
            return

        class MockEvent:
             def __init__(self, button, xdata, ydata, inaxes):
                  self.button = button
                  self.xdata = xdata
                  self.ydata = ydata
                  self.inaxes = inaxes

        if event.xdata is not None and event.ydata is not None:
             col = int(round(event.xdata))
             row = int(round(event.ydata))
             if 0 <= row < self.map_editor.height and 0 <= col < self.map_editor.width:
                  mock_event = MockEvent(event.button, event.xdata, event.ydata, event.inaxes)
                  self.map_editor.onclick(mock_event)
                  self.update_visualization(show_paths=True)

    def create_new_map(self):
        width = self.width_spin.value()
        height = self.height_spin.value()
        env_type_vn = self.env_combo.currentText().lower()
        env_type_map = {
            "đồng nhất": "uniform",
            "kho hàng": "warehouse",
            "thành phố": "city"
        }
        env_type = env_type_map.get(env_type_vn, "uniform")
        obstacle_prob = self.obstacle_spin.value()
        num_targets = self.targets_spin.value()
        
        self.map_editor = InteractiveMapEditor(width=width, height=height)
        self.map_editor.randomize(
            obstacle_prob=obstacle_prob,
            num_targets=num_targets,
            env_type=env_type
        )
        
        temp_file = "temp_map.npy"
        map_data = {
            "grid": self.map_editor.grid,
            "start": self.map_editor.start,
            "goal": self.map_editor.goal,
            "targets": self.map_editor.targets
        }
        np.save(temp_file, map_data)
        self.current_map_file = temp_file
        
        self.paths = None
        self.visited_nodes_all_algos = None
        self.orders = None
        self.path_lines = {} # Xóa các đường đi cũ

        self.update_visualization()
        self.order_label.setText("")

        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(False)

    def load_map(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Tải bản đồ", "", "NumPy Files (*.npy)"
        )
        if file_name:
            try:
                data = np.load(file_name, allow_pickle=True).item()
                width, height = data["grid"].shape
                
                self.map_editor = InteractiveMapEditor(width=width, height=height)
                self.map_editor.grid = data["grid"]
                self.map_editor.start = data["start"]
                self.map_editor.goal = data["goal"]
                self.map_editor.targets = data["targets"]
                
                self.current_map_file = file_name

                self.paths = None
                self.visited_nodes_all_algos = None
                self.orders = None
                self.path_lines = {} 

                self.update_visualization()
                self.order_label.setText("")

                for checkbox in self.path_checkboxes.values():
                     checkbox.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Tải bản đồ thất bại: {str(e)}")
                
    def save_map(self):
        if not self.map_editor:
            QMessageBox.warning(self, "Cảnh Báo", "Không có bản đồ để lưu!")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Lưu Bản Đồ", "", "NumPy Files (*.npy)"
        )
        if file_name:
            try:
                map_data = {
                    "grid": self.map_editor.grid,
                    "start": self.map_editor.start,
                    "goal": self.map_editor.goal,
                    "targets": self.map_editor.targets
                }
                np.save(file_name, map_data)
                self.current_map_file = file_name
                QMessageBox.information(self, "Thành Công", "Đã lưu bản đồ thành công!")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Lưu bản đồ thất bại: {str(e)}")
                
    def run_algorithm(self):
        if not self.current_map_file:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng tải hoặc tạo bản đồ trước!")
            return
            
        try:
            selected_algo_option = self.algo_combo.currentText()
            
            if selected_algo_option == "Tất Cả Thuật Toán":
                result = run_algorithms(self.current_map_file)
                
                if result is None or not isinstance(result, tuple) or len(result) != 4:
                    QMessageBox.critical(self, "Lỗi", "Chạy thuật toán thất bại. Vui lòng kiểm tra lại cài đặt bản đồ.")
                    return
                    
                paths, visited_nodes_raw, orders, metrics = result

                self.paths = paths
                self.visited_nodes_all_algos = visited_nodes_raw
                self.orders = orders
                
                self.update_comparison_table(metrics)
                
                for name, checkbox in self.path_checkboxes.items():
                    checkbox.setEnabled(True)
                    checkbox.setChecked(True)
            else:
                result = run_algorithms(self.current_map_file, selected_algo_option)
                
                if result is None or not isinstance(result, tuple) or len(result) != 4:
                     QMessageBox.critical(self, "Lỗi", f"Chạy {selected_algo_option} thất bại. Vui lòng kiểm tra lại cài đặt bản đồ.")
                     return
                     
                paths, visited_nodes_raw, orders, metrics = result

                self.paths = {selected_algo_option: paths.get(selected_algo_option)}
                self.visited_nodes_all_algos = {selected_algo_option: visited_nodes_raw.get(selected_algo_option)}
                self.orders = {selected_algo_option: orders.get(selected_algo_option)}
                
                self.update_comparison_table({selected_algo_option: metrics.get(selected_algo_option)})
                
                for name, checkbox in self.path_checkboxes.items():
                    checkbox.setEnabled(name == selected_algo_option)
                    checkbox.setChecked(name == selected_algo_option)

            
            self.update_visualization(show_paths=True)
            
           
            self.update_order_label()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Chạy thuật toán thất bại: {str(e)}")

    def toggle_path_visibility(self, algo_name, checked):
        if algo_name in self.path_lines and self.path_lines[algo_name]:
            self.path_lines[algo_name].set_visible(checked)
            self.update_visualization(show_paths=True)
            self.update_order_label() 

    def update_comparison_table(self, metrics_data):
        """Cập nhật bảng so sánh với dữ liệu thống kê."""
        
        if not metrics_data:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            self.comparison_table.clear()
            self.update_order_label()
            return

        
        all_metric_names = set()
        for algo_metrics in metrics_data.values():
            if isinstance(algo_metrics, dict):
                all_metric_names.update(algo_metrics.keys())

        

        metric_row_headers = sorted(list(all_metric_names))

        
        has_order_data = self.orders and any(isinstance(order, list) and order for order in self.orders.values())

        if "Target Order" in metric_row_headers:
             metric_row_headers.remove("Target Order")

        
        if has_order_data:
            metric_row_headers.append("Thứ Tự Mục Tiêu")

        metric_row_headers.append("Điểm Tổng Hợp")

        
        display_names = []
        for metric_name in metric_row_headers:
            if metric_name == "Length":
                display_names.append("Độ Dài (m)")
            elif metric_name == "Inflect.":
                display_names.append("Số Lần Rẽ (lần)")
            elif metric_name == "Angle":
                display_names.append("Tổng Góc Rẽ (°)")
            elif metric_name == "Nodes":
                display_names.append("Số Node Duyệt (node)")
            elif metric_name == "Thứ Tự Mục Tiêu":
                display_names.append("Thứ Tự Mục Tiêu")
            elif metric_name == "Điểm Tổng Hợp":
                display_names.append("Điểm Tổng Hợp")
            else:
                display_names.append(metric_name)

        self.comparison_table.setRowCount(len(metric_row_headers))
        self.comparison_table.setColumnCount(len(self.algorithm_names))

        
        self.comparison_table.setVerticalHeaderLabels(display_names)
        self.comparison_table.setHorizontalHeaderLabels(self.algorithm_names)

        
        header_to_row_idx = {header: i for i, header in enumerate(metric_row_headers)}

        
        for col_idx, algo_name in enumerate(self.algorithm_names):
            algo_metrics = metrics_data.get(algo_name, {})
            if not isinstance(algo_metrics, dict):
                algo_metrics = {}
            for metric_name in metric_row_headers:
                if metric_name == "Điểm Tổng Hợp":
                    continue  
                row_idx = header_to_row_idx.get(metric_name)
                if row_idx is not None:
                    if metric_name == "Thứ Tự Mục Tiêu":
                        order = self.orders.get(algo_name)
                        if order is not None and self.map_editor and self.map_editor.targets is not None:
                             value = self.format_single_order(order, self.map_editor.targets)
                        else:
                             value = "N/A"
                        item = QTableWidgetItem(value)
                        self.comparison_table.setItem(row_idx, col_idx, item)
                    else:
                        value = algo_metrics.get(metric_name, "N/A")
                        if isinstance(value, (int, float)):
                            if metric_name == "Length":
                                value_meters = value * 1.0  
                                value = f"{value_meters:.1f} m"
                            elif metric_name == "Inflect.":
                                value = f"{value:.0f}"
                            elif metric_name == "Angle":
                                value = f"{value:.1f}°"
                            elif metric_name == "Nodes":
                                value = f"{value:.0f}"
                            else:
                                value = str(value)
                        else:
                            value = str(value)
                        item = QTableWidgetItem(value)
                        self.comparison_table.setItem(row_idx, col_idx, item)    
        overall_scores = calculate_overall_scores(metrics_data)
        row_idx = header_to_row_idx.get("Điểm Tổng Hợp")
        for col_idx, algo_name in enumerate(self.algorithm_names):
            score = overall_scores.get(algo_name, 0.0)
            item = QTableWidgetItem(f"{score:.3f}")
            self.comparison_table.setItem(row_idx, col_idx, item)
        self.comparison_table.resizeColumnsToContents()

    def update_visualization(self, show_paths=False, visited_nodes_highlight=None, algo_to_highlight=None):
        if not self.map_editor:
            return
        self.ax.clear()
        
        grid = self.map_editor.grid
        height, width = grid.shape
        self.ax.imshow(grid, cmap='binary', origin='upper', aspect='equal',
                      extent=[-0.5, width-0.5, height-0.5, -0.5])

        self.ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xlim(-0.5, width-0.5)
        self.ax.set_ylim(height-0.5, -0.5)
        
        if self.map_editor.start:
            sx, sy = self.map_editor.start
            self.ax.add_patch(plt.Rectangle((sy - 0.5, sx - 0.5), 1, 1, color=(0, 0.8, 0), label='Điểm Xuất Phát'))
        if self.map_editor.goal:
            gx, gy = self.map_editor.goal
            self.ax.add_patch(plt.Rectangle((gy - 0.5, gx - 0.5), 1, 1, color=(0.8, 0, 0), label='Điểm Đích'))
        for i, target in enumerate(self.map_editor.targets):
            tx, ty = target
            self.ax.add_patch(plt.Rectangle((ty - 0.5, tx - 0.5), 1, 1, color=(1, 1, 0), label='Mục Tiêu' if i == 0 else ""))
            self.ax.text(ty, tx, str(i+1), color='blue', ha='center', va='center', fontsize=8, fontweight='bold')
            
        if visited_nodes_highlight:
            visited_array = np.array(visited_nodes_highlight)
            if visited_array.size > 0:
                self.ax.scatter(visited_array[:, 1], visited_array[:, 0], color='skyblue', s=20, label='Các Node Đã Duyệt', zorder=5)

        if show_paths and self.paths:
            colors = ['red', 'blue', 'purple', 'green']
            self.path_lines = {}

            for i, name in enumerate(self.algorithm_names):
                path = self.paths.get(name)
                if path:
                    path_array = np.array(path)
                    if name in self.path_checkboxes and self.path_checkboxes[name].isChecked():
                        line, = self.ax.plot(path_array[:, 1], path_array[:, 0], color=colors[i % len(colors)], linewidth=2, label=f'{name} ({i+1})', zorder=4)
                        self.path_lines[name] = line
                        line.set_visible(True)
                    else:
                        line, = self.ax.plot([], [], color=colors[i % len(colors)], linewidth=2, label=f'{name} ({i+1})', zorder=4)
                        self.path_lines[name] = line
                        line.set_visible(False)

        
        if show_paths or visited_nodes_highlight:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        self.ax.set_xticks(np.arange(-.5, self.map_editor.grid.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.map_editor.grid.shape[0], 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_title("Bản đồ tìm đường")

        self.canvas.draw()

    def update_order_label(self, algo_name=None):
        """Cập nhật nhãn thứ tự mục tiêu bên dưới bảng dựa vào các đường đi đang hiển thị hoặc thuật toán cụ thể."""
        if not self.orders or not self.map_editor or not self.map_editor.targets:
            self.order_label.setText("")
            return

        current_targets = self.map_editor.targets

        order_texts = []
        if algo_name and algo_name != "Tất Cả Thuật Toán":
            
            order = self.orders.get(algo_name)
            if order is not None:
                formatted_order_text = self.format_single_order(order, current_targets)
                if formatted_order_text:
                     order_texts.append(f"{algo_name}: {formatted_order_text}")
        else:
            
            for key_name in self.algorithm_names:
                 order = self.orders.get(key_name)
                 if order is not None and self.path_checkboxes.get(key_name, QCheckBox()).isChecked():
                     formatted_order_text = self.format_single_order(order, current_targets)
                     if formatted_order_text:
                         order_texts.append(f"{key_name}: {formatted_order_text}")
        self.order_label.setText("")

    def format_single_order(self, order, targets):
        """Hàm hỗ trợ định dạng thứ tự đi qua các mục tiêu."""
        full_sequence_elements = ["S"]
        if isinstance(order, list) and targets:
            ordered_target_names = []
            for t_idx in order:
                if isinstance(t_idx, int) and 0 <= t_idx < len(targets):
                    ordered_target_names.append(str(t_idx + 1))
            full_sequence_elements.extend(ordered_target_names)
        full_sequence_elements.append("G")
        return ' → '.join(full_sequence_elements)

    def eventFilter(self, obj, event):
        """Lọc sự kiện để xử lý phím tắt bàn phím."""
        if event.type() == event.KeyPress:
            key = event.key()
            key_text = None
            if Qt.Key_1 <= key <= Qt.Key_5:
                key_text = str(key - Qt.Key_0)
            elif key == Qt.Key_Space:
                key_text = ' '
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                key_text = 'enter'
            if key_text:
                 self.on_key_press(key_text)
                 return True
        return super().eventFilter(obj, event)

    def on_key_press(self, key_text):
        """Xử lý phím tắt bàn phím."""
        print(f"Đã nhấn phím: {key_text}")
        if key_text.isdigit():
            index = int(key_text) - 1
            if 0 <= index < len(self.algorithm_names):
                algo_name = self.algorithm_names[index]
                if algo_name in self.path_checkboxes:
                    checkbox = self.path_checkboxes[algo_name]
                    checkbox.setChecked(not checkbox.isChecked())
        elif key_text == ' ':
            self.create_new_map()
        elif key_text == 'enter':
            self.save_map()

    def edit_map_manually(self):
        """Mở trình chỉnh sửa bản đồ thủ công."""
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.map_editor = InteractiveMapEditor(width=width, height=height)
        self.map_editor.run()
        self.update_visualization()
        temp_file = "temp_map.npy"
        map_data = {
            "grid": self.map_editor.grid,
            "start": self.map_editor.start,
            "goal": self.map_editor.goal,
            "targets": self.map_editor.targets
        }
        np.save(temp_file, map_data)
        self.current_map_file = temp_file
        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(True)
            checkbox.setChecked(True)
        self.paths = None
        self.visited_nodes_all_algos = None
        self.orders = None
        self.path_lines = {}

    def duplicate_map(self):
        if not self.map_editor:
            QMessageBox.warning(self, "Cảnh báo", "Chưa có bản đồ để sao chép!")
            return
        new_editor = InteractiveMapEditor(
            width=self.map_editor.width,
            height=self.map_editor.height
        )
        new_editor.grid = np.copy(self.map_editor.grid)
        new_editor.start = self.map_editor.start
        new_editor.goal = self.map_editor.goal
        new_editor.targets = list(self.map_editor.targets)
        self.map_editor = new_editor
        self.current_map_file = None  
        self.map_editor.run()         
        self.update_visualization()
        QMessageBox.information(self, "Thành công", "Đã tạo bản đồ mới dựa trên bản đồ hiện tại. Bạn có thể chỉnh sửa và lưu lại với tên mới.")

def main():
    app = QApplication(sys.argv)
    window = PathfindingGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 
