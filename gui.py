import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
                            QGroupBox, QGridLayout, QCheckBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from main import run_algorithms
from map_utils import InteractiveMapEditor
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# ...
# gui.py

# TẠO LỚP TOOLBAR TÙY CHỈNH NÀY
class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)

    def back(self, *args, **kwargs):
        super().back(*args, **kwargs)
        self.canvas.draw_idle()

class PathfindingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pathfinding Algorithms Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5) # Further reduce spacing between group boxes and widgets
        
        # Map creation/loading section
        map_group = QGroupBox("Map Settings")
        map_layout = QVBoxLayout()
        
        # Map size controls
        size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 100)
        self.width_spin.setValue(20)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 100)
        self.height_spin.setValue(20)
        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.height_spin)

        # Add Number of Targets to the same row
        self.targets_spin = QSpinBox() # Keep initialization here
        self.targets_spin.setRange(1, 10)
        self.targets_spin.setValue(1)
        size_layout.addSpacing(20) # Add some space
        size_layout.addWidget(QLabel("Number of Targets:"))
        size_layout.addWidget(self.targets_spin)
        size_layout.addStretch() # Push everything to the left

        map_layout.addLayout(size_layout) # Add this horizontal layout
        
        # Environment type selection
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "Uniform",
            "Warehouse",
            "City"
        ])
        
        # Connect environment type change to show/hide cluster count
        self.env_combo.currentTextChanged.connect(self.on_environment_type_changed)
        
        # Obstacle probability
        self.obstacle_spin = QDoubleSpinBox()
        self.obstacle_spin.setRange(0, 0.4)
        self.obstacle_spin.setValue(0.2)
        self.obstacle_spin.setSingleStep(0.1)
        
        # Create a horizontal layout for Environment Type and Obstacle Probability
        env_obstacle_layout = QHBoxLayout()
        env_obstacle_layout.addWidget(QLabel("Environment Type:"))
        env_obstacle_layout.addWidget(self.env_combo, stretch=1) # Give stretch to combo box
        env_obstacle_layout.addSpacing(20) # Add some space between the two controls
        self.obstacle_label = QLabel("Obstacle Probability:")
        env_obstacle_layout.addWidget(self.obstacle_label)
        env_obstacle_layout.addWidget(self.obstacle_spin)
        env_obstacle_layout.addStretch() # Push everything to the left

        # Add the new horizontal layout to the main map layout
        map_layout.addLayout(env_obstacle_layout) # Add the second horizontal layout
        
        # Map control buttons
        button_layout = QHBoxLayout()
        self.create_btn = QPushButton("Create New Map")
        self.load_btn = QPushButton("Load Map")
        self.save_btn = QPushButton("Save Map")
        self.edit_btn = QPushButton("Edit Map Manually")  # Add new button
        button_layout.addWidget(self.create_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.edit_btn)  # Add to layout
        map_layout.addLayout(button_layout)
        
        map_group.setLayout(map_layout)
        left_layout.addWidget(map_group)
        
        # Algorithm selection section
        algo_group = QGroupBox("Algorithm Settings")
        algo_layout = QVBoxLayout()
        # Reduce spacing between widgets in this layout to 0
        algo_layout.setSpacing(0)
        algo_layout.setContentsMargins(5, 0, 5, 0) # Reduce top and bottom margins to 0
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "All Algorithms",
            "A* truyền thống",
            "A* truyền thống + Greedy",
            "A* cải tiến",
            "A* cải tiến + Greedy",
            "A* cải tiến + ACO"
        ])

        # Create a horizontal layout for the algorithm selection
        algo_select_layout = QHBoxLayout()
        algo_select_layout.addWidget(QLabel("Select Algorithm:"))
        algo_select_layout.addWidget(self.algo_combo, stretch=1) # Give stretch to the combo box
        algo_select_layout.addStretch() # Push to the left

        # Add the new horizontal layout to the main algo layout
        algo_layout.addLayout(algo_select_layout)
        
        self.run_btn = QPushButton("Run Algorithm")
        algo_layout.addWidget(self.run_btn)
        
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)
        
        # Path visibility section
        visibility_group = QGroupBox("Path Visibility")
        visibility_layout = QVBoxLayout()

        self.path_checkboxes = {}
        self.path_lines = {}

        # These names must match the keys returned by run_algorithms in main.py
        self.algorithm_names = [
            "A* truyền thống",
            "A* truyền thống + Greedy",
            "A* cải tiến",
            "A* cải tiến + Greedy",
            "A* cải tiến + ACO"
        ]

        for name in self.algorithm_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True) # Paths visible by default
            checkbox.toggled.connect(lambda checked, algo=name: self.toggle_path_visibility(algo, checked))
            visibility_layout.addWidget(checkbox)
            self.path_checkboxes[name] = checkbox

        visibility_group.setLayout(visibility_layout)
        left_layout.addWidget(visibility_group)

        # Label for target order (Moved from left panel)
        self.order_label = QLabel("")
        self.order_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.order_label.setStyleSheet("color: red;")
        self.order_label.setWordWrap(True)
        left_layout.addWidget(self.order_label)

        # Add left panel to main layout
        layout.addWidget(left_panel, stretch=1)
        
        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Matplotlib figure for visualization
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111) # Create axes once
        
        # Add navigation toolbar for zoom and pan
        self.toolbar = CustomToolbar(self.canvas, right_panel)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1) # Give canvas stretch

        # Comparison Table section (Moved from left panel)
        compare_group = QGroupBox("Algorithm Comparison")
        compare_layout = QVBoxLayout()

        self.comparison_table = QTableWidget()

        compare_layout.addWidget(self.comparison_table)
        compare_group.setLayout(compare_layout)
        right_layout.addWidget(compare_group, stretch=0)

        # Add right panel to main layout
        layout.addWidget(right_panel, stretch=5)
        
        # Connect signals
        self.create_btn.clicked.connect(self.create_new_map)
        self.load_btn.clicked.connect(self.load_map)
        self.save_btn.clicked.connect(self.save_map)
        self.run_btn.clicked.connect(self.run_algorithm)
        self.edit_btn.clicked.connect(self.edit_map_manually)  # Connect new button

        self.current_map_file = None
        self.map_editor = None
        self.paths = None
        self.visited_nodes_all_algos = None
        self.orders = None

        # Connect map interaction events
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        
        # Connect keyboard events to the main window
        self.installEventFilter(self)

        # Disable path visibility checkboxes initially
        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(False)

    def on_environment_type_changed(self, env_type):
        # Hide Obstacle Probability for Warehouse and City
        if env_type.lower() in ["warehouse", "city"]:
            self.obstacle_spin.hide()
            self.obstacle_label.hide()
        else:
            self.obstacle_spin.show()
            self.obstacle_label.show()
        
    def on_map_click(self, event):
        """Handle mouse clicks on the Matplotlib canvas for map editing."""
        if not self.map_editor or event.inaxes != self.figure.gca():
            return

        # Pass the click event to the map editor's onclick method
        # We need to create a mock event object that has the attributes map_editor expects
        class MockEvent:
             def __init__(self, button, xdata, ydata, inaxes):
                  self.button = button
                  self.xdata = xdata
                  self.ydata = ydata
                  self.inaxes = inaxes

        # Check if the click was within the grid bounds before creating the mock event
        if event.xdata is not None and event.ydata is not None:
             col = int(round(event.xdata))
             row = int(round(event.ydata))
             if 0 <= row < self.map_editor.height and 0 <= col < self.map_editor.width:
                  mock_event = MockEvent(event.button, event.xdata, event.ydata, event.inaxes)
                  self.map_editor.onclick(mock_event)
                  # After editing, update the visualization to show changes
                  self.update_visualization(show_paths=True)

    def create_new_map(self):
        width = self.width_spin.value()
        height = self.height_spin.value()
        env_type = self.env_combo.currentText().lower()
        obstacle_prob = self.obstacle_spin.value()
        num_targets = self.targets_spin.value()
        
        self.map_editor = InteractiveMapEditor(width=width, height=height)
        self.map_editor.randomize(
            obstacle_prob=obstacle_prob,
            num_targets=num_targets,
            env_type=env_type
        )
        
        # Save the map to a temporary file
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
        self.path_lines = {} # Clear previous path lines

        self.update_visualization()
        self.order_label.setText("")

        # Disable path visibility checkboxes on new map
        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(False)

    def load_map(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Map", "", "NumPy Files (*.npy)"
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
                self.path_lines = {} # Clear previous path lines

                self.update_visualization()
                self.order_label.setText("")

                # Disable path visibility checkboxes on load
                for checkbox in self.path_checkboxes.values():
                     checkbox.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load map: {str(e)}")
                
    def save_map(self):
        if not self.map_editor:
            QMessageBox.warning(self, "Warning", "No map to save!")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Map", "", "NumPy Files (*.npy)"
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
                QMessageBox.information(self, "Success", "Map saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save map: {str(e)}")
                
    def run_algorithm(self):
        if not self.current_map_file:
            QMessageBox.warning(self, "Warning", "Please load or create a map first!")
            return
            
        try:
            selected_algo_option = self.algo_combo.currentText()
            
            # Run algorithms based on selection
            if selected_algo_option == "All Algorithms":
                # Run all algorithms
                result = run_algorithms(self.current_map_file)
                
                # Check if algorithms ran successfully
                if result is None or not isinstance(result, tuple) or len(result) != 4:
                    QMessageBox.critical(self, "Error", "Failed to run algorithms. Please check map settings.")
                    return
                    
                paths, visited_nodes_raw, orders, metrics = result

                self.paths = paths
                self.visited_nodes_all_algos = visited_nodes_raw
                self.orders = orders
                
                # Update comparison table with all metrics
                self.update_comparison_table(metrics)
                
                # Enable and check all checkboxes
                for name, checkbox in self.path_checkboxes.items():
                    checkbox.setEnabled(True)
                    checkbox.setChecked(True)
            else:
                # Run only the selected algorithm
                result = run_algorithms(self.current_map_file, selected_algo_option)
                
                # Check if algorithm ran successfully
                if result is None or not isinstance(result, tuple) or len(result) != 4:
                     QMessageBox.critical(self, "Error", f"Failed to run {selected_algo_option}. Please check map settings.")
                     return
                     
                paths, visited_nodes_raw, orders, metrics = result

                # Note: When running a single algorithm, metrics and orders will only contain data for that algorithm.
                # GUI logic handles this by only displaying available data.

                self.paths = {selected_algo_option: paths.get(selected_algo_option)}
                self.visited_nodes_all_algos = {selected_algo_option: visited_nodes_raw.get(selected_algo_option)}
                self.orders = {selected_algo_option: orders.get(selected_algo_option)}
                
                # Update comparison table with only the selected algorithm's metrics
                self.update_comparison_table({selected_algo_option: metrics.get(selected_algo_option)})
                
                # Disable all checkboxes except the selected one
                for name, checkbox in self.path_checkboxes.items():
                    checkbox.setEnabled(name == selected_algo_option)
                    checkbox.setChecked(name == selected_algo_option)

            # Update visualization to show final paths based on checkbox states
            self.update_visualization(show_paths=True)
            
            # Update order label based on visible paths
            self.update_order_label()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run algorithm: {str(e)}")

    def toggle_path_visibility(self, algo_name, checked):
        if algo_name in self.path_lines and self.path_lines[algo_name]:
            self.path_lines[algo_name].set_visible(checked)
            # After toggling visibility, update the order label to reflect visible paths
            # Ensure update_order_label is called here
            self.update_visualization(show_paths=True)
            self.update_order_label() 

    def update_comparison_table(self, metrics_data):
        """Populate the comparison table with metrics data."""
        
        if not metrics_data:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            self.comparison_table.clear()
            self.update_order_label()
            return

        # Get all metric names from the metrics data
        all_metric_names = set()
        for algo_metrics in metrics_data.values():
            if isinstance(algo_metrics, dict):
                all_metric_names.update(algo_metrics.keys())

        # Sort metric names for consistent display
        metric_row_headers = sorted(list(all_metric_names))

        # Ensure 'Collision' is the last row if present
        has_collision_data = "Collision" in metric_row_headers
        if has_collision_data:
            metric_row_headers.remove("Collision")

        # Ensure 'Target Order' is the second to last row if present and we actually have order data
        # Check if self.orders is not None and has any non-empty list values
        has_order_data = self.orders and any(isinstance(order, list) and order for order in self.orders.values())

        if "Target Order" in metric_row_headers:
             metric_row_headers.remove("Target Order")

        # Only add 'Target Order' to headers if there is actual order data to show
        if has_order_data:
            metric_row_headers.append("Target Order")

        # Add 'Collision' back at the end if it was present
        if has_collision_data:
            metric_row_headers.append("Collision")

        # Set up table dimensions
        self.comparison_table.setRowCount(len(metric_row_headers))
        self.comparison_table.setColumnCount(len(self.algorithm_names))

        # Set headers
        self.comparison_table.setVerticalHeaderLabels(metric_row_headers)
        self.comparison_table.setHorizontalHeaderLabels(self.algorithm_names)

        # Create mapping of header to row index
        header_to_row_idx = {header: i for i, header in enumerate(metric_row_headers)}

        # Populate table cells
        for col_idx, algo_name in enumerate(self.algorithm_names):
            algo_metrics = metrics_data.get(algo_name, {})
            # Ensure algo_metrics is a dictionary
            if not isinstance(algo_metrics, dict):
                algo_metrics = {}
            

            # Populate metric rows
            for metric_name in metric_row_headers:
                row_idx = header_to_row_idx.get(metric_name)
                if row_idx is not None:
                    if metric_name == "Target Order":
                        # Handle Target Order separately using the calculated order from self.orders
                        order = self.orders.get(algo_name)
                        
                        # self.map_editor and self.map_editor.targets must be valid to format
                        if order is not None and self.map_editor and self.map_editor.targets is not None:
                             # format_single_order already handles empty order list by returning S->G
                             value = self.format_single_order(order, self.map_editor.targets)
                             
                        else:
                             # Display placeholder if order data or map data is missing
                             value = "N/A"
                             

                        # Create and set the item
                        item = QTableWidgetItem(value)
                        self.comparison_table.setItem(row_idx, col_idx, item)

                    else:
                        # Handle other metrics
                        value = algo_metrics.get(metric_name, "N/A")
                        if isinstance(value, (int, float)):
                            if metric_name in ["Runtime (s)", "Planning (s)"]:
                                value = f"{value:.3f}"
                            elif metric_name in ["Angle", "Length"]:
                                value = f"{value:.1f}"
                            else:
                                value = str(value)
                        else:
                            value = str(value)

                        # Create and set the item for other metrics
                        item = QTableWidgetItem(value)
                        self.comparison_table.setItem(row_idx, col_idx, item)

        # Auto-resize columns for better display
        self.comparison_table.resizeColumnsToContents()

    def update_visualization(self, show_paths=False, visited_nodes_highlight=None, algo_to_highlight=None):
        if not self.map_editor:
            return
            
        # Clear everything except the base map and features (start, goal, targets)
        self.ax.clear()
        
        # Plot the grid
        grid = self.map_editor.grid
        self.ax.imshow(grid, cmap='binary', origin='upper', aspect='equal') # Set aspect to equal
        
        # Explicitly set axes limits and aspect to ensure square cells
        self.ax.set_xlim(-0.5, grid.shape[1] - 0.5)
        self.ax.set_ylim(grid.shape[0] - 0.5, -0.5) # Reverse y-axis for 'upper' origin
        self.ax.set_aspect('equal', adjustable='box') # Set axes aspect to equal
        
        # Plot start, goal, and targets
        if self.map_editor.start:
            self.ax.plot(self.map_editor.start[1], self.map_editor.start[0], 'go', markersize=10, label='Start')
        if self.map_editor.goal:
            self.ax.plot(self.map_editor.goal[1], self.map_editor.goal[0], 'ro', markersize=10, label='Goal')
        for i, target in enumerate(self.map_editor.targets):
            self.ax.plot(target[1], target[0], 'bo', markersize=8)
            self.ax.text(target[1], target[0], str(i+1), color='white', ha='center', va='center', fontsize=8)
            
        # Highlight visited nodes during animation
        if visited_nodes_highlight:
            visited_array = np.array(visited_nodes_highlight)
            if visited_array.size > 0:
                self.ax.scatter(visited_array[:, 1], visited_array[:, 0], color='skyblue', s=20, label='Visited Nodes', zorder=5)

        # Plot final paths if show_paths is True and we have paths data
        if show_paths and self.paths:
            colors = ['red', 'orange', 'blue', 'purple', 'green']
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

        # Add legend outside the plot
        if show_paths or visited_nodes_highlight:
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set grid boundaries and title
        self.ax.set_xticks(np.arange(-.5, self.map_editor.grid.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.map_editor.grid.shape[0], 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_title("Pathfinding Map")

        # Adjust plot to make space for the legend and order text
        self.figure.tight_layout(rect=[0, 0.1, 0.85, 1])  # Increased bottom margin for order text

        self.canvas.draw()

    def update_order_label(self, algo_name=None):
        """Update the order label below the table based on visible paths or a specific algo."""
        # This QLabel is now primarily for showing order during animation or specific selections
        # The main comparison table has the order info after running algorithms.

        if not self.orders or not self.map_editor or not self.map_editor.targets:
            self.order_label.setText("")
            return

        current_targets = self.map_editor.targets

        order_texts = []
        if algo_name and algo_name != "All Algorithms":
            # Show order for a specific algorithm (e.g., during animation)
            order = self.orders.get(algo_name)
            # Ensure format_single_order is called correctly and handles None order
            # Only format if order data is available
            if order is not None:
                formatted_order_text = self.format_single_order(order, current_targets)
                # Display even if only S->G (empty order list) to show sequence
                if formatted_order_text:
                     order_texts.append(f"{algo_name}: {formatted_order_text}")
        else:
            # Show order for all algorithms with checked paths
            for key_name in self.algorithm_names:
                 order = self.orders.get(key_name)
                 # Only format if order data is available and checkbox is checked
                 if order is not None and self.path_checkboxes.get(key_name, QCheckBox()).isChecked():
                     formatted_order_text = self.format_single_order(order, current_targets)
                     # Display even if only S->G (empty order list) to show sequence
                     if formatted_order_text:
                         order_texts.append(f"{key_name}: {formatted_order_text}")

        # Update the QLabel text
        self.order_label.setText("")

    # Define format_single_order as a class method
    def format_single_order(self, order, targets):
        """Helper to format the target visit order."""
        # Start and Goal are always part of the full sequence
        full_sequence_elements = ["S"]

        # Check if order is a valid list and targets are available
        if isinstance(order, list) and targets:
            # Map target indices in the order list to their 1-based numbers
            # Filter out any invalid indices just in case
            # Ensure we only process indices that correspond to actual targets
            ordered_target_names = []
            for t_idx in order:
                if isinstance(t_idx, int) and 0 <= t_idx < len(targets):
                    ordered_target_names.append(str(t_idx + 1))
                # Optional: Add a warning here if unexpected data is in the order list
                # else:
                # print(f"Warning: Unexpected data in order list: {t_idx}")

            full_sequence_elements.extend(ordered_target_names)

        # Append Goal at the end
        full_sequence_elements.append("G")

        # The list full_sequence_elements now contains the elements in order (S, target numbers, G)
        # Join them with the arrow separator
        return ' → '.join(full_sequence_elements)

    def eventFilter(self, obj, event):
        """Filter events to handle keyboard shortcuts."""
        if event.type() == event.KeyPress:
            key = event.key()
            # Map Qt key codes to string representation
            key_text = None
            if Qt.Key_1 <= key <= Qt.Key_5:
                key_text = str(key - Qt.Key_0)
            elif key == Qt.Key_Space:
                key_text = ' '
            elif key == Qt.Key_Return or key == Qt.Key_Enter:
                key_text = 'enter'
                
            if key_text:
                 self.on_key_press(key_text)
                 return True # Event handled

        return super().eventFilter(obj, event)

    def on_key_press(self, key_text):
        """Handle keyboard presses."""
        print(f"Key pressed: {key_text}") # For debugging

        if key_text.isdigit(): # Keys '1' through '5'
            index = int(key_text) - 1
            if 0 <= index < len(self.algorithm_names):
                algo_name = self.algorithm_names[index]
                if algo_name in self.path_checkboxes:
                    checkbox = self.path_checkboxes[algo_name]
                    # Toggle checkbox state, which triggers toggle_path_visibility
                    checkbox.setChecked(not checkbox.isChecked())

        elif key_text == ' ': # Space key
            self.create_new_map()

        elif key_text == 'enter': # Enter key
            self.save_map()

    def edit_map_manually(self):
        """Open the interactive map editor for manual editing."""
        # Always create a new map editor with current width and height
        width = self.width_spin.value()
        height = self.height_spin.value()
        self.map_editor = InteractiveMapEditor(width=width, height=height)
            
        # Run the editor
        self.map_editor.run()
        
        # After editing, update the visualization
        self.update_visualization()
        
        # Save the edited map to a temporary file for algorithm running
        temp_file = "temp_map.npy"
        map_data = {
            "grid": self.map_editor.grid,
            "start": self.map_editor.start,
            "goal": self.map_editor.goal,
            "targets": self.map_editor.targets
        }
        np.save(temp_file, map_data)
        self.current_map_file = temp_file

        # Enable path visibility checkboxes after editing
        for checkbox in self.path_checkboxes.values():
            checkbox.setEnabled(True)
            checkbox.setChecked(True)

        # Clear previous paths and results
        self.paths = None
        self.visited_nodes_all_algos = None
        self.orders = None
        self.path_lines = {}

def main():
    app = QApplication(sys.argv)
    window = PathfindingGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 