import matplotlib.pyplot as plt
import numpy as np

def plot_grid_map_compare(grid_map, start, goal,
                          path_traditional=None,
                          path_improved=None,
                          path_improved2=None,
                          path_improved3=None,
                          targets=None,
                          title="So sánh các đường đi"):
    height, width = grid_map.shape
    fig, ax = plt.subplots(figsize=(width / 2, height / 2))
    ax.set_title(title)
    lines = {}

    # ==== Vẽ chướng ngại vật ====
    for x in range(height):
        for y in range(width):
            if grid_map[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, height - x - 1), 1, 1, color='black'))

    # ==== Vẽ Start, Goal, Targets ====
    sx, sy = start
    ax.add_patch(plt.Rectangle((sy, height - sx - 1), 1, 1, color='green', label='Start'))

    gx, gy = goal
    ax.add_patch(plt.Rectangle((gy, height - gx - 1), 1, 1, color='red', label='Goal'))

    if targets:
        for tx, ty in targets:
            ax.add_patch(plt.Rectangle((ty, height - tx - 1), 1, 1, color='orange', label='Target'))

    # ==== Hàm vẽ đường đi có thể ẩn/hiện ====
    def draw_path(path, color, label, keycode):
        if not path:
            return []
        segments = []
        # dummy line để vào legend
        dummy, = ax.plot([], [], color=color, linewidth=2, label=label)
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            line, = ax.plot(
                [y1 + 0.5, y2 + 0.5],
                [height - x1 - 0.5, height - x2 - 0.5],
                color=color, linewidth=2, visible=False
            )
            segments.append(line)
        lines[keycode] = segments

    # ==== Vẽ các đường đi ====
    draw_path(path_traditional, 'red', "A* truyền thống (1)", keycode="1")
    draw_path(path_improved, 'green', "A* TT + Greedy (2)", keycode="2")
    draw_path(path_improved2, 'blue', "A* cải tiến (3)", keycode="3")
    draw_path(path_improved3, 'violet', "A* CT + Greedy (4)", keycode="4")

    # ==== Thiết lập grid ====
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    ax.grid(True)

    # ==== Chú thích ====
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10)

    # ==== Tương tác phím ====
    def on_key(event):
        if event.key in lines:
            for line in lines[event.key]:
                line.set_visible(not line.get_visible())
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
