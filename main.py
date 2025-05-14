import os
import time
import math
import numpy as np
from tabulate import tabulate
from map_utils import InteractiveMapEditor
from astar_trad import astar_with_targets
from astar_imp import  astar_improved_with_targets
from astar_imp_with_greedy import astar_improved_with_targets_greedy
from astar_trad_with_greedy import astar_with_greedy_targets
from visualization import plot_grid_map_compare

def count_inflections_and_turn_angle(path):
    if len(path) < 3:
        return 0, 0.0
    def direction(a, b):
        return b[0] - a[0], b[1] - a[1]
    inflections = 0
    total_angle = 0.0
    for i in range(1, len(path) - 1):
        v1 = direction(path[i - 1], path[i])
        v2 = direction(path[i], path[i + 1])
        if v1 != v2:
            inflections += 1
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            if len1 > 0 and len2 > 0:
                angle = math.acos(max(-1, min(1, dot / (len1 * len2))))
                total_angle += math.degrees(angle)
    return inflections, total_angle

def total_path_length(path):
    return sum(math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path)-1))

def compute_path_stats(path, visited, runtime, planning_time):
    if not path:
        return None, None, None, None, None, None
    length = total_path_length(path)
    inflections, angle = count_inflections_and_turn_angle(path)
    return planning_time, runtime, len(visited), inflections, angle, length

def safe_format(value, fmt):
    return fmt.format(value) if value is not None else "-"

# ======== CHỌN HOẶC TẠO BẢN ĐỒ =========
saved_maps = [f for f in os.listdir() if f.endswith(".npy")]
if saved_maps:
    print("\n📂 Danh sách bản đồ đã lưu:")
    for i, f in enumerate(saved_maps):
        print(f"{i + 1}. {f}")
    choice = input("🔢 Chọn bản đồ (số) hoặc nhập tên mới: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(saved_maps):
        map_file = saved_maps[int(choice) - 1]
    else:
        map_file = f"{choice}.npy"
        w = int(input("🔧 Nhập width bản đồ: "))
        h = int(input("🔧 Nhập height bản đồ: "))
        InteractiveMapEditor(width=w, height=h).run()
else:
    print("⚠️ Không có bản đồ nào, tạo mới...")
    map_file = "map1.npy"
    w = int(input("🔧 Nhập width bản đồ: "))
    h = int(input("🔧 Nhập height bản đồ: "))
    InteractiveMapEditor(width=w, height=h).run()

data = np.load(map_file, allow_pickle=True).item()
grid = data.get("grid")
start = data.get("start")
goal = data.get("goal")
targets = data.get("targets", [])

if start is None or goal is None:
    print("❌ Thiếu Start hoặc Goal!")
    exit()

# ======== CHẠY THUẬT TOÁN =========

print("▶️ A* truyền thống...")
t0 = time.time()
path1, visited1 = astar_with_targets(grid, start, targets, goal, return_visited=True)
runtime1 = time.time() - t0
stat1 = compute_path_stats(path1, visited1, runtime1, runtime1)

print("🤖 A* truyền thống + Greedy...")
t1 = time.time()
path2, visited2 = astar_with_greedy_targets(grid, start, targets, goal, return_visited=True)
runtime2 = time.time() - t1
stat2 = compute_path_stats(path2, visited2, runtime2, runtime2)

print("✨ A* cải tiến...")
t2 = time.time()
path3, visited3 = astar_improved_with_targets(grid, start, targets, goal)
runtime3 = time.time() - t2
stat3 = compute_path_stats(path3, visited3, runtime3, runtime3)

print("💡 A* cải tiến + Greedy...")
t3 = time.time()
path4, visited4 = astar_improved_with_targets_greedy(grid, start, targets, goal)
runtime4 = time.time() - t3
stat4 = compute_path_stats(path4, visited4, runtime4, runtime4)

# ======== IN BẢNG THỐNG KÊ =========

stats = [
    #["Run time (s)",           safe_format(stat1[1], "{:.4f}"), safe_format(stat2[1], "{:.4f}"),
    #                           safe_format(stat3[1], "{:.4f}"), safe_format(stat4[1], "{:.4f}")],
    ["Number of nodes",        stat1[2], stat2[2], stat3[2], stat4[2]],
    ["Inflection points",      stat1[3], stat2[3], stat3[3], stat4[3]],
    ["Total turning angle (°)", safe_format(stat1[4], "{:.2f}"), safe_format(stat2[4], "{:.2f}"),
                               safe_format(stat3[4], "{:.2f}"), safe_format(stat4[4], "{:.2f}")],
    ["Path length",            safe_format(stat1[5], "{:.2f}"), safe_format(stat2[5], "{:.2f}"),
                               safe_format(stat3[5], "{:.2f}"), safe_format(stat4[5], "{:.2f}")]
]

print("\n📊 SO SÁNH THUẬT TOÁN")
print(tabulate(stats, headers=["Thông số", "A* TT", "A* TT + Greedy", "A* CT", "A* CT + Greedy"], tablefmt="fancy_grid"))

# ======== VẼ BẢN ĐỒ TƯƠNG TÁC =========
print("🎨 Nhấn phím [1-4] để hiện/ẩn các đường đi:")
print("  [1] A* TT - đỏ")
print("  [2] A* TT + Greedy - xanh lá")
print("  [3] A* CT + Greedy - xanh dương")
print("  [4] A* CT - tím")
plot_grid_map_compare(
    grid, start, goal,
    path1, path2, path3, path4,
    targets
)
