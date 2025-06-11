import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox

# Cấu hình matplotlib để chỉ hiển thị cửa sổ chính
plt.rcParams['figure.figsize'] = [12, 10]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.max_open_warning'] = 1
plt.rcParams['figure.raise_window'] = True

class InteractiveMapEditor:
    def __init__(self, width=30, height=30, map_file="saved_map.npy"):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.targets = []
        self.map_file = map_file
        
        # Đóng tất cả các cửa sổ hiện có
        plt.close('all')
        
        # Khởi tạo figure và axes với kích thước lớn hơn
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Interactive Map Editor')
        
        # Tạo grid layout
        gs = self.fig.add_gridspec(2, 1, height_ratios=[0.9, 0.1])
        
        # Axes chính cho bản đồ
        self.ax = self.fig.add_subplot(gs[0])
        
        # Axes cho status bar
        self.status_ax = self.fig.add_subplot(gs[1])
        self.status_ax.axis('off')
        
        # Tạo image plot cho lưới
        self.grid_plot = self.ax.imshow(self.grid, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        
        # Thiết lập các sự kiện
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Thiết lập trục
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        self.ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Thêm tiêu đề
        self.ax.set_title('Interactive Map Editor', pad=20, fontsize=14, fontweight='bold')
        
        # Thêm chú thích
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Start'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Goal'),
            plt.Rectangle((0, 0), 1, 1, facecolor='orange', label='Target')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Vẽ lưới ban đầu
        self.draw_grid()
        
        # Cập nhật status bar
        self.update_status()

    def update_status(self):
        self.status_ax.clear()
        self.status_ax.axis('off')
        status_text = (
            f"Status: {'Ready' if self.start and self.goal else 'Setting up...'} | "
            f"Start: {self.start if self.start else 'Not set'} | "
            f"Goal: {self.goal if self.goal else 'Not set'} | "
            f"Targets: {len(self.targets)}"
        )
        self.status_ax.text(0.02, 0.5, status_text, fontsize=10, va='center')
        self.fig.canvas.draw_idle()

    def draw_grid(self):
        # Xóa tất cả các phần tử hiện có trên axes
        self.ax.clear()
        
        # Thiết lập lại các thuộc tính cơ bản
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        self.ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Tạo ma trận màu cho lưới
        grid_colors = np.ones((self.height, self.width, 3), dtype=float)  # Mặc định màu trắng
        
        # Đặt màu cho các ô có chướng ngại vật
        grid_colors[self.grid == 1] = [0.5, 0.5, 0.5]  # Xám cho chướng ngại vật
        
        # Đặt màu cho start, goal và targets
        if self.start:
            grid_colors[self.start[0], self.start[1]] = [0, 0.8, 0]  # Xanh lá cho start
        if self.goal:
            grid_colors[self.goal[0], self.goal[1]] = [0.8, 0, 0]  # Đỏ cho goal
        for tx, ty in self.targets:
            grid_colors[tx, ty] = [1, 0.5, 0]  # Cam cho targets
            
        # Vẽ lại grid
        self.grid_plot = self.ax.imshow(grid_colors, origin='upper', extent=[-0.5, self.width-0.5, self.height-0.5, -0.5])
        
        # Thêm tiêu đề
        self.ax.set_title('Interactive Map Editor', pad=20, fontsize=14, fontweight='bold')
        
        # Thêm chú thích
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Start'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Goal'),
            plt.Rectangle((0, 0), 1, 1, facecolor='orange', label='Target')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Cập nhật status bar
        self.update_status()
        
        # Vẽ lại figure
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
            
        # Chuyển đổi tọa độ chuột thành tọa độ lưới
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        
        if not (0 <= row < self.height and 0 <= col < self.width):
            return

        if event.button == 1:  # Left click – toggle obstacle
            if (row, col) not in [self.start, self.goal] + self.targets:
                self.grid[row, col] = 1 - self.grid[row, col]

        elif event.button == 3:  # Right click – set start, goal, target
            if self.start is None:
                self.start = (row, col)
                self.grid[row, col] = 0  # Loại bỏ obstacle nếu có
            elif self.goal is None and (row, col) != self.start:
                self.goal = (row, col)
                self.grid[row, col] = 0
            elif (row, col) not in [self.start, self.goal] and (row, col) not in self.targets:
                self.targets.append((row, col))
                self.grid[row, col] = 0  # Loại bỏ obstacle nếu có

        self.draw_grid()


    def onkeypress(self, event):
        if event.key == 'enter':
            print("✔ Hoàn tất tạo bản đồ!")
            print("Start:", self.start)
            print("Goal:", self.goal)
            print("Targets:", self.targets)
            map_data = {
                "grid": self.grid,
                "start": self.start,
                "goal": self.goal,
                "targets": self.targets
            }
            np.save(self.map_file, map_data)
            print(f"💾 Đã lưu bản đồ vào file '{self.map_file}'")
            plt.close()

        elif event.key == 'backspace':
            if self.targets:
                self.targets.pop()
            elif self.goal:
                self.goal = None
            elif self.start:
                self.start = None
            self.draw_grid()
            
        elif event.key == ' ':  # Space key
            if hasattr(self, 'current_env_type'):
                print("🔄 Đang tạo lại môi trường...")
                self.randomize(
                    obstacle_prob=self.current_obstacle_prob,
                    num_targets=self.current_num_targets,
                    env_type=self.current_env_type
                )

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.2
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
            
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
            
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_xlim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_xlim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n🎨 Interactive Map Editor")
        print("=" * 50)
        print("Controls:")
        print("🖱 Left Click: Toggle obstacle")
        print("🖱 Right Click: Set Start → Goal → Targets")
        print("⌫ Backspace: Remove Target/Goal/Start")
        print("⏎ Enter: Save and exit")
        print("🖱 Mouse Wheel: Zoom in/out")
        print("␣ Space: Randomize environment")
        print("=" * 50)
        
        # Hiển thị cửa sổ và đảm bảo nó ở trên cùng
        self.fig.canvas.manager.window.raise_()
        plt.show(block=True)

    def is_path_possible(self, start, goal, grid):
        """Kiểm tra xem có đường đi từ start đến goal không (phiên bản tối ưu)"""
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return False
            
        # Sử dụng BFS đơn giản hơn, chỉ kiểm tra 4 hướng
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            if current == goal:
                return True
                
            # Chỉ kiểm tra 4 hướng cơ bản
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                if (0 <= next_x < self.height and 
                    0 <= next_y < self.width and 
                    grid[next_x, next_y] == 0 and 
                    (next_x, next_y) not in visited):
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))
        return False

    def ensure_connectivity(self, grid):
        """Đảm bảo môi trường có tính liên thông (phiên bản tối ưu)"""
        # Tạo các lối đi ngẫu nhiên để đảm bảo tính liên thông
        num_paths = int(self.height * self.width * 0.1)  # 10% số ô
        for _ in range(num_paths):
            x = np.random.randint(1, self.height-1)
            y = np.random.randint(1, self.width-1)
            if grid[x, y] == 1:
                # Tạo lối đi 2x2
                grid[x-1:x+1, y-1:y+1] = 0
        return grid

    def randomize(self, obstacle_prob=0.2, num_targets=1, env_type='default', num_clusters=None):
        # Lưu các tham số hiện tại để có thể tạo lại môi trường
        self.current_obstacle_prob = obstacle_prob
        self.current_num_targets = num_targets
        self.current_env_type = env_type
        
        max_attempts = 5  # Giảm số lần thử xuống 5
        for attempt in range(max_attempts):
            self.grid = np.zeros((self.height, self.width), dtype=int)

            if env_type == 'default':
                self.grid = (np.random.rand(self.height, self.width) < obstacle_prob).astype(int)

            elif env_type == 'uniform':
                # Môi trường ngẫu nhiên đơn giản
                self.grid = (np.random.rand(self.height, self.width) < obstacle_prob).astype(int)

            elif env_type == 'warehouse':
                # Mô phỏng kệ hàng: tạo các dải tường thẳng với độ ngẫu nhiên
                self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 1
                for i in range(2, self.height - 2, 4):
                    if np.random.random() < 0.8:
                        shelf_width = np.random.randint(1, 3)
                        for w in range(shelf_width):
                            if i + w < self.height - 2:
                                for j in range(1, self.width - 1):
                                    if np.random.random() > 0.1:
                                        self.grid[i + w, j] = 1
                # Tạo lối đi giữa các kệ
                for i in range(3, self.height - 3, 4):
                    if np.random.random() < 0.3:
                        self.grid[i, :] = 0

            elif env_type == 'city':
                # Mô phỏng thành phố: đường lưới với độ ngẫu nhiên
                road_spacing = np.random.randint(4, 7)
                for i in range(0, self.height, road_spacing):
                    road_width = np.random.randint(1, 3)
                    for w in range(road_width):
                        if i + w < self.height:
                            self.grid[i + w, :] = 1
                for j in range(0, self.width, road_spacing):
                    road_width = np.random.randint(1, 3)
                    for w in range(road_width):
                        if j + w < self.width:
                            self.grid[:, j + w] = 1
                # Tạo lối đi giữa các tòa nhà
                for i in range(road_spacing, self.height, road_spacing):
                    for j in range(road_spacing, self.width, road_spacing):
                        if np.random.random() < 0.3:
                            self.grid[i-1:i+2, j-1:j+2] = 0

            else:
                raise ValueError(f"Loại môi trường không hợp lệ: {env_type}")

            # --- Logic for placing S/G/T and checking path connectivity for *other* env types ---
            # This part needs to be done *after* the grid is generated for the specific type.
            # This applies ONLY IF env_type IS NOT 'guaranteed_path'
            # Select free cells for start/goal/targets
            free_cells = list(zip(*np.where(self.grid == 0)))
            np.random.shuffle(free_cells)

            # Check if there are enough free cells
            required_free_cells = 2 + num_targets
            if len(free_cells) < required_free_cells:
                print(f"⚠️ Không đủ ô trống ({len(free_cells)}) để đặt Start, Goal và {num_targets} Targets. Cần {required_free_cells}. Thử lại.")
                self.grid = np.zeros((self.height, self.width), dtype=int)
                self.start = None
                continue  # Try again (next attempt)

            # Attempt to place Start, Goal, Targets
            try:
                self.start = free_cells.pop()
                self.goal = free_cells.pop()
                self.targets = [free_cells.pop() for _ in range(num_targets)]

                # Check path connectivity (Start-Goal and involving targets)
                valid_path_found = self.check_all_path_connectivity(self.grid, self.start, self.goal, self.targets)

                if valid_path_found:
                    break # Exit attempt loop
                else:
                    print("⚠️ Không có đường đi hợp lệ trên bản đồ ngẫu nhiên đã tạo. Thử lại.")
                    # Path not found, continue to the next attempt
                    continue

            except IndexError:
                # Not enough free cells left after placing some points
                print("⚠️ Hết ô trống khi đặt Start, Goal, Targets. Thử lại.")
                continue # Go to the next attempt

        # If the loop finished without generating a valid map in any attempt
        if not valid_path_found:
             print("❌ Không thể tạo bản đồ hợp lệ sau nhiều lần thử!")
             # For now, just draw whatever the last attempt resulted in (might be empty or invalid)

        self.draw_grid()

    def check_all_path_connectivity(self, grid, start, goal, targets):
        """Checks connectivity from start to all targets, between all targets, and from targets to goal."""
        points_to_check = [start] + targets + [goal]
        # Check connectivity between all pairs of (start, targets, goal)
        # A more robust check for TSP would be needed for order, but this checks basic reachability.
        
        # Create a temporary grid for checking paths without modifying the main grid
        temp_grid = np.copy(grid)

        # Check connectivity using the simple BFS (4 directions)
        # Need to ensure is_path_possible handles targets correctly, currently it only checks S-G.
        # Let's reuse is_path_possible but call it multiple times.

        # Check Start to Goal
        if not self.is_path_possible(start, goal, temp_grid):
            return False
            
        # Check Start to all Targets
        for target in targets:
            if not self.is_path_possible(start, target, temp_grid):
                return False

        # Check connectivity between all pairs of Targets (if more than one)
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if not self.is_path_possible(targets[i], targets[j], temp_grid):
                    return False
                    
        # Check all Targets to Goal
        for target in targets:
            if not self.is_path_possible(target, goal, temp_grid):
                return False

        # If all checks pass
        return True

    def on_environment_type_changed(self, env_type):
        if env_type.lower() in ["warehouse", "city"]:
            self.obstacle_spin.hide()
            self.obstacle_label.hide()
        else:
            self.obstacle_spin.show()
            self.obstacle_label.show()
            if env_type.lower() == "mountain":
                self.obstacle_spin.setMaximum(0.3)  # Giới hạn tối đa 0.3
                if self.obstacle_spin.value() > 0.3:
                    self.obstacle_spin.setValue(0.3)
            else:
                self.obstacle_spin.setMaximum(1.0)

if __name__ == "__main__":
    editor = InteractiveMapEditor(width=20, height=20)
    editor.run()
