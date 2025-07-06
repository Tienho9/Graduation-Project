import matplotlib.pyplot as plt
import numpy as np

class InteractiveMapEditor:
    def __init__(self, width=30, height=30, map_file="saved_map.npy"):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.targets = []
        self.map_file = map_file
        
        plt.close('all')
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Interactive Map Editor')
        gs = self.fig.add_gridspec(2, 1, height_ratios=[0.9, 0.1])
        
        self.ax = self.fig.add_subplot(gs[0])
        
        self.status_ax = self.fig.add_subplot(gs[1])
        self.status_ax.axis('off')
        
        self.grid_plot = self.ax.imshow(self.grid, cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        self.ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
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
        self.ax.clear()
        self.ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        self.ax.grid(True, which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        grid_colors = np.ones((self.height, self.width, 3), dtype=float)  
        grid_colors[self.grid == 1] = [0.5, 0.5, 0.5]  
        if self.start:
            grid_colors[self.start[0], self.start[1]] = [0, 0.8, 0]  
        if self.goal:
            grid_colors[self.goal[0], self.goal[1]] = [0.8, 0, 0]  
        for tx, ty in self.targets:
            grid_colors[tx, ty] = [1, 0.5, 0]  
            
        self.grid_plot = self.ax.imshow(grid_colors, origin='upper', extent=[-0.5, self.width-0.5, self.height-0.5, -0.5])
        
        self.ax.set_title('Interactive Map Editor', pad=20, fontsize=14, fontweight='bold')
    
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='gray', label='Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Start'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Goal'),
            plt.Rectangle((0, 0), 1, 1, facecolor='orange', label='Target')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        self.update_status()
        
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        
        if not (0 <= row < self.height and 0 <= col < self.width):
            return

        pos = (row, col)

        if event.button == 1:  
            if self.start == pos:
                self.start = None
            elif self.goal == pos:
                self.goal = None
            elif pos in self.targets:
                self.targets.remove(pos)
            else:
                self.grid[row, col] = 1 - self.grid[row, col]

        elif event.button == 3:  
            if self.start is None:
                self.start = pos
                self.grid[row, col] = 0  
            elif self.goal is None and pos != self.start:
                self.goal = pos
                self.grid[row, col] = 0
            elif pos not in [self.start, self.goal] and pos not in self.targets:
                self.targets.append(pos)
                self.grid[row, col] = 0  

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
            
        elif event.key == ' ':  
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
        
        self.fig.canvas.manager.window.raise_()
        plt.show(block=True)

    def is_path_possible(self, start, goal, grid):
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return False
            
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            if current == goal:
                return True
                
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
        num_paths = int(self.height * self.width * 0.1)  
        for _ in range(num_paths):
            x = np.random.randint(1, self.height-1)
            y = np.random.randint(1, self.width-1)
            if grid[x, y] == 1:
                grid[x-1:x+1, y-1:y+1] = 0
        return grid

    def randomize(self, obstacle_prob=0.2, num_targets=1, env_type='default', num_clusters=None):
        self.current_obstacle_prob = obstacle_prob
        self.current_num_targets = num_targets
        self.current_env_type = env_type
        
        attempts_count = 0
        max_total_attempts = 100 

        while True:
            attempts_count += 1
            if attempts_count > max_total_attempts:
                print(f"❌ KHÔNG THỂ TẠO BẢN ĐỒ HỢP LỆ sau {max_total_attempts} lần thử!")
                print("Vui lòng giảm mật độ chướng ngại vật hoặc tăng kích thước bản đồ.")
                self.grid = np.zeros((self.height, self.width), dtype=int)
                self.start = None
                self.goal = None
                self.targets = []
                break 

            self.grid = np.zeros((self.height, self.width), dtype=int)

            if env_type == 'default':
                self.grid = (np.random.rand(self.height, self.width) < obstacle_prob).astype(int)

            elif env_type == 'uniform':
                self.grid = (np.random.rand(self.height, self.width) < obstacle_prob).astype(int)

            elif env_type == 'warehouse':
                self.grid[0, :] = self.grid[-1, :] = self.grid[:, 0] = self.grid[:, -1] = 1
                for i in range(2, self.height - 2, 4):
                    if np.random.random() < 0.8:
                        shelf_width = np.random.randint(1, 3)
                        for w in range(shelf_width):
                            if i + w < self.height - 2:
                                for j in range(1, self.width - 1):
                                    if np.random.random() > 0.1:
                                        self.grid[i + w, j] = 1
                for i in range(3, self.height - 3, 4):
                    if np.random.random() < 0.3:
                        self.grid[i, :] = 0

            elif env_type == 'city':
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
                for i in range(road_spacing, self.height, road_spacing):
                    for j in range(road_spacing, self.width, road_spacing):
                        if np.random.random() < 0.3:
                            self.grid[i-1:i+2, j-1:j+2] = 0
            elif env_type == 'organic':
                num_shapes = np.random.randint(5, 15)
                for _ in range(num_shapes):
                    shape_type = np.random.choice(['circle', 'rect', 'blob'])
                    cx = np.random.randint(3, self.height - 3)
                    cy = np.random.randint(3, self.width - 3)
                    size = np.random.randint(2, 5)

                    if shape_type == 'circle':
                        for dx in range(-size, size+1):
                            for dy in range(-size, size+1):
                                if dx**2 + dy**2 <= size**2:
                                    x, y = cx + dx, cy + dy
                                    if 0 <= x < self.height and 0 <= y < self.width:
                                        self.grid[x, y] = 1

                    elif shape_type == 'rect':
                        h = np.random.randint(2, 5)
                        w = np.random.randint(2, 5)
                        for dx in range(h):
                            for dy in range(w):
                                x, y = cx + dx, cy + dy
                                if 0 <= x < self.height and 0 <= y < self.width:
                                    self.grid[x, y] = 1

                    elif shape_type == 'blob':
                        points = [(cx, cy)]
                        for _ in range(size * 3):
                            px, py = points[-1]
                            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                                if np.random.rand() < 0.5:
                                    nx, ny = px + dx, py + dy
                                    if 0 <= nx < self.height and 0 <= ny < self.width:
                                        self.grid[nx, ny] = 1
                                        points.append((nx, ny))

            else:
                raise ValueError(f"Loại môi trường không hợp lệ: {env_type}")

            free_cells = list(zip(*np.where(self.grid == 0)))
            np.random.shuffle(free_cells)

            required_free_cells = 2 + num_targets
            if len(free_cells) < required_free_cells:
                continue

            try:
                self.start = free_cells.pop()
                self.goal = free_cells.pop()
                self.targets = [free_cells.pop() for _ in range(num_targets)]

                valid_path_found = self.check_all_path_connectivity(self.grid, self.start, self.goal, self.targets)

                if valid_path_found:
                    print(f"✅ Đã tạo bản đồ hợp lệ sau {attempts_count} lần thử.")
                    break 
                else:
                    continue 

            except IndexError:
                continue

        self.draw_grid()

    def check_all_path_connectivity(self, grid, start, goal, targets):
        points_to_check = [start] + targets + [goal]
        temp_grid = np.copy(grid)

        if not self.is_path_possible(start, goal, temp_grid):
            return False
            
        for target in targets:
            if not self.is_path_possible(start, target, temp_grid):
                return False

        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if not self.is_path_possible(targets[i], targets[j], temp_grid):
                    return False
                    
        for target in targets:
            if not self.is_path_possible(target, goal, temp_grid):
                return False

        return True

    def on_environment_type_changed(self, env_type):
        if env_type.lower() in ["warehouse", "city"]:
            self.obstacle_spin.hide()
            self.obstacle_label.hide()
        else:
            self.obstacle_spin.show()
            self.obstacle_label.show()
            if env_type.lower() == "mountain":
                self.obstacle_spin.setMaximum(0.3)  
                if self.obstacle_spin.value() > 0.3:
                    self.obstacle_spin.setValue(0.3)
            else:
                self.obstacle_spin.setMaximum(1.0)

if __name__ == "__main__":
    editor = InteractiveMapEditor(width=20, height=20)
    editor.run()
