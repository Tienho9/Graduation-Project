import matplotlib.pyplot as plt
import numpy as np

def plot_grid_map_compare(grid_map, start, goal,
                          path_traditional=None,
                          path_greedy=None,
                          path_improved=None,
                          path_improved_greedy=None,
                          path_aco=None,
                          targets=None,
                          title="So sánh các đường đi",
                          targets_ordered_list=None):
    height, width = grid_map.shape
    fig, ax = plt.subplots(figsize=(width / 2, height / 2))
    ax.set_title(title)
    lines = {}
    order_texts = {}
    order_labels = {
        "1": "A* TT",
        "2": "TT+Greedy",
        "3": "CT",
        "4": "CT+Greedy",
        "5": "CT+ACO"
    }

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

    # Hiển thị thứ tự target khi khởi tạo (T1, T2, ...)
    if targets:
        for idx, (tx, ty) in enumerate(targets):
            ax.add_patch(plt.Rectangle((ty, height - tx - 1), 1, 1, color='orange', label='Target' if idx == 0 else ""))
            ax.text(ty + 0.5, height - tx - 0.5, f"T{idx+1}", color='blue', fontsize=7, ha='center', va='center', fontweight='bold')

    # ==== Hàm vẽ đường đi có thể ẩn/hiện ====
    def draw_path(path, color, label, keycode):
        if path is None:
            # Add a dummy line for legend even if path is empty
            dummy, = ax.plot([], [], color=color, linewidth=2, label=f'{label} ({keycode})')
            lines[keycode] = [] # Store an empty list if no path
            order_texts[keycode] = None # No order text if no path
            return

        segments = []
        # dummy line để vào legend
        dummy, = ax.plot([], [], color=color, linewidth=2, label=f'{label} ({keycode})')
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            line, = ax.plot(
                [y1 + 0.8, y2 + 0.8],
                [height - x1 - 0.6, height - x2 - 0.6],
                color=color, linewidth=2, visible=False # Paths are initially hidden
            )
            segments.append(line)
        lines[keycode] = segments # Store segments for this keycode

        # Draw points on the path (excluding start/goal/targets)
        for node in path:
            if node != start and node != goal and node not in targets:
                point, = ax.plot(node[1] + 0.8, height - node[0] - 0.6, '.', color=color, markersize=4, visible=False)
                segments.append(point) # Include points in segments to toggle visibility

    # ==== Vẽ các đường đi ====
    draw_path(path_traditional, 'red', "A* truyền thống", keycode="1")
    draw_path(path_greedy, 'blue', "A* truyền thống + Greedy", keycode="2")
    draw_path(path_improved, 'green', "A* cải tiến", keycode="3")
    draw_path(path_improved_greedy, 'purple', "A* cải tiến + Greedy", keycode="4")
    draw_path(path_aco, 'orange', "A* cải tiến + ACO", keycode="5")

    # ==== Thiết lập grid ====
    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.grid(True)

    # ==== Chú thích ====
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Place legend outside the plot area to the right
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10)

    # ==== Tương tác phím ====
    def on_key(event):
        # Handle visibility for path segments
        if event.key in lines:
            for line in lines[event.key]:
                line.set_visible(not line.get_visible())
        # Redraw canvas
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    return fig
