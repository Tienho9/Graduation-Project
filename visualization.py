import matplotlib.pyplot as plt
import numpy as np

def plot_grid_map_compare(grid_map, start, goal,
                          path_traditional=None,
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
        "1": "A*",
        "2": "ImpA*",
        "3": "ImpA*G",
        "4": "ImpA*ACO"
    }

    for x in range(height):
        for y in range(width):
            if grid_map[x][y] == 1:
                ax.add_patch(plt.Rectangle((y, height - x - 1), 1, 1, color='black'))

    sx, sy = start
    ax.add_patch(plt.Rectangle((sy, height - sx - 1), 1, 1, color='green', label='Start'))

    gx, gy = goal
    ax.add_patch(plt.Rectangle((gy, height - gx - 1), 1, 1, color='red', label='Goal'))

    if targets:
        for idx, (tx, ty) in enumerate(targets):
            ax.add_patch(plt.Rectangle((ty, height - tx - 1), 1, 1, color='orange', label='Target' if idx == 0 else ""))
            ax.text(ty + 0.5, height - tx - 0.5, f"T{idx+1}", color='blue', fontsize=7, ha='center', va='center', fontweight='bold')

    def draw_path(path, color, label, keycode):
        if path is None:
            dummy, = ax.plot([], [], color=color, linewidth=2, label=f'{label} ({keycode})')
            lines[keycode] = [] 
            order_texts[keycode] = None 
            return

        segments = []
        dummy, = ax.plot([], [], color=color, linewidth=2, label=f'{label} ({keycode})')
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            line, = ax.plot(
                [y1 + 0.6, y2 + 0.6],
                [height - x1 - 0.5, height - x2 - 0.5],
                color=color, linewidth=2, visible=False 
            )
            segments.append(line)
        lines[keycode] = segments 

        for node in path:
            if node != start and node != goal and node not in targets:
                point, = ax.plot(node[1] + 0.6, height - node[0] - 0.5, '.', color=color, markersize=4, visible=False)
                segments.append(point) 
    draw_path(path_traditional, 'red', "A*", keycode="1")
    draw_path(path_improved, 'blue', "ImpA*", keycode="2")
    draw_path(path_improved_greedy, 'purple', "ImpA*G", keycode="3")
    draw_path(path_aco, 'green', "ImpA*ACO", keycode="4")

    ax.set_xticks(np.arange(0, width, 1))
    ax.set_yticks(np.arange(0, height, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10)

    def on_key(event):
        if event.key in lines:
            for line in lines[event.key]:
                line.set_visible(not line.get_visible())
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    return fig
