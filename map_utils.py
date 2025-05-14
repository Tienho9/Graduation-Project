import matplotlib.pyplot as plt
import numpy as np

class InteractiveMapEditor:
    def __init__(self, width=30, height=30):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.targets = []

        self.fig, self.ax = plt.subplots()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
        self.draw_grid()

    def draw_grid(self):
        self.ax.clear()
        for x in range(self.height):
            for y in range(self.width):
                value = self.grid[x, y]
                color = 'white'
                if value == 1:
                    color = 'black'
                self.ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color=color))

        if self.start:
            x, y = self.start
            self.ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='green'))
        if self.goal:
            x, y = self.goal
            self.ax.add_patch(plt.Rectangle((y, self.height - x - 1), 1, 1, color='red'))
        for tx, ty in self.targets:
            self.ax.add_patch(plt.Rectangle((ty, self.height - tx - 1), 1, 1, color='orange'))

        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks(np.arange(0, self.width, 1))
        self.ax.set_yticks(np.arange(0, self.height, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        col = int(event.xdata)
        row = self.height - int(event.ydata) - 1

        if event.button == 1:  # Left click – toggle obstacle
            if (row, col) not in [self.start, self.goal] + self.targets:
                self.grid[row, col] = 1 - self.grid[row, col]

        elif event.button == 3:  # Right click – set start, goal, target
            if self.start is None:
                self.start = (row, col)
            elif self.goal is None and (row, col) != self.start:
                self.goal = (row, col)
            elif (row, col) not in [self.start, self.goal] and (row, col) not in self.targets:
                self.targets.append((row, col))

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
            np.save("saved_map.npy", map_data)
            print("💾 Đã lưu bản đồ vào file 'saved_map.npy'")
            plt.close()

        elif event.key == 'backspace':
            if self.targets:
                self.targets.pop()
            elif self.goal:
                self.goal = None
            elif self.start:
                self.start = None
            self.draw_grid()

    def run(self):
        print("🎨 Tạo bản đồ:")
        print("🖱 Trái: bật/tắt chướng ngại vật")
        print("🖱 Phải: chọn Start → Goal → các Target")
        print("⌫ Backspace: xóa Target/Goal/Start")
        print("⏎ Enter: kết thúc và lưu bản đồ")
        plt.show()

if __name__ == "__main__":
    editor = InteractiveMapEditor(width=20, height=20)
    editor.run()
