import os
import time
import math
import numpy as np
from tabulate import tabulate
from map_utils import InteractiveMapEditor
from astar_trad import astar_with_targets
from astar_imp import astar_improved_with_targets
from astar_imp_with_greedy import astar_improved_with_targets_greedy
from astar_trad_with_greedy import astar_with_greedy_targets
from visualization import plot_grid_map_compare
from astar_imp_with_aco import astar_improved_with_targets_aco

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

def run_algorithms(map_file):
    data = np.load(map_file, allow_pickle=True).item()
    grid = data.get("grid")
    start = data.get("start")
    goal = data.get("goal")
    targets = data.get("targets", [])

    if start is None or goal is None:
        print("❌ Thiếu Start hoặc Goal!")
        return

    print("▶️ A* truyền thống...")
    t0 = time.time()
    path1, visited1, order1 = astar_with_targets(grid, start, targets, goal, return_visited=True)
    runtime1 = time.time() - t0
    stat1 = compute_path_stats(path1, visited1, runtime1, runtime1)

    print("🤖 A* truyền thống + Greedy...")
    t1 = time.time()
    path2, visited2, order2 = astar_with_greedy_targets(grid, start, targets, goal, return_visited=True)
    runtime2 = time.time() - t1
    stat2 = compute_path_stats(path2, visited2, runtime2, runtime2)

    print("✨ A* cải tiến...")
    t2 = time.time()
    path3, visited3, order3 = astar_improved_with_targets(grid, start, targets, goal)
    runtime3 = time.time() - t2
    stat3 = compute_path_stats(path3, visited3, runtime3, runtime3)

    print("💡 A* cải tiến + Greedy...")
    t3 = time.time()
    path4, visited4, order4 = astar_improved_with_targets_greedy(grid, start, targets, goal)
    runtime4 = time.time() - t3
    stat4 = compute_path_stats(path4, visited4, runtime4, runtime4)

    print("🌟 A* cải tiến + ACO...")
    t6 = time.time()
    path6, visited6, order6 = astar_improved_with_targets_aco(grid, start, targets, goal)
    runtime6 = time.time() - t6
    stat6 = compute_path_stats(path6, visited6, runtime6, runtime6)

    stats = [
        ["Nodes",        stat1[2], stat2[2], stat3[2], stat4[2], stat6[2]],
        ["Inflect.",      stat1[3], stat2[3], stat3[3], stat4[3], stat6[3]],
        ["Angle", safe_format(stat1[4], "{:.1f}"), safe_format(stat2[4], "{:.1f}"),
                               safe_format(stat3[4], "{:.1f}"), safe_format(stat4[4], "{:.1f}"), 
                               safe_format(stat6[4], "{:.1f}")],
        ["Length",            safe_format(stat1[5], "{:.1f}"), safe_format(stat2[5], "{:.1f}"),
                               safe_format(stat3[5], "{:.1f}"), safe_format(stat4[5], "{:.1f}"), 
                               safe_format(stat6[5], "{:.1f}")]
    ]

    print("\n📊 SO SÁNH THUẬT TOÁN")
    print(tabulate(stats, headers=["Info", "A*TT", "TT+G", "CT", "CT+G", "CT+ACO"], tablefmt="fancy_grid"))

    # Tạo text thứ tự target đi qua cho từng thuật toán
    def order_text(order):
        if not order or not targets:
            return "Không có target nào."
        return ' → '.join([f"T{i+1}" for i in order])

    import matplotlib.pyplot as plt
    print("🎨 Nhấn phím [1-5] để hiện/ẩn các đường đi:")
    fig = plot_grid_map_compare(
        grid, start, goal,
        path1, path2, path3, path4, path6,
        targets,
        targets_ordered_list=[order1, order2, order3, order4, order6]
    )
    # Hiển thị thứ tự dưới lưới
    plt.figtext(0.5, 0.01, f"A* TT: {order_text(order1)} | TT+Greedy: {order_text(order2)} | CT: {order_text(order3)} | CT+Greedy: {order_text(order4)} | CT+ACO: {order_text(order6)}", ha='center', fontsize=11, color='red')
    plt.show()

# ============================ TẠO HOẶC CHỌN BẢN ĐỒ ===============================

saved_maps = [f for f in os.listdir() if f.endswith(".npy")]
if saved_maps:
    print("\n📂 Danh sách bản đồ đã lưu:")
    for i, f in enumerate(saved_maps):
        print(f"{i + 1}. {f}")
    choice = input("🔢 Chọn bản đồ (số) hoặc nhập tên mới: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(saved_maps):
        map_file = saved_maps[int(choice) - 1]
        # Kiểm tra và load file bản đồ an toàn
        try:
            map_data = np.load(map_file, allow_pickle=True).item()
            if not all(k in map_data for k in ["grid", "start", "goal", "targets"]):
                print(f"❌ File '{map_file}' thiếu trường dữ liệu cần thiết (grid, start, goal, targets). Hãy tạo lại bản đồ!")
                exit(1)
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file '{map_file}'. Hãy kiểm tra lại tên file!")
            exit(1)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file '{map_file}': {e}. Hãy kiểm tra lại file!")
            exit(1)
        run_algorithms(map_file)
    else:
        map_file = f"{choice}.npy"
        w = int(input("🔧 Nhập width bản đồ: "))
        h = int(input("🔧 Nhập height bản đồ: "))
        random_choice = input("🌱 Sinh ngẫu nhiên bản đồ? (Y/n): ").strip().lower()
        editor = InteractiveMapEditor(width=w, height=h, map_file=map_file)
        if random_choice == '' or random_choice == 'y':
            try:
                print("\n🌍 Chọn loại môi trường:")
                print("1. Mặc định (ngẫu nhiên)")
                print("2. Nhà kho (warehouse)")
                print("3. Thành phố (city)")
                print("4. Mê cung (maze)")
                print("5. Rừng (forest)")
                print("6. Núi (mountain)")
                env_choice = input("🔢 Chọn loại môi trường (1-6, mặc định 1): ").strip()
                env_types = {
                    '1': 'default',
                    '2': 'warehouse',
                    '3': 'city',
                    '4': 'maze',
                    '5': 'forest',
                    '6': 'mountain'
                }
                env_type = env_types.get(env_choice, 'default')
                
                obstacle_prob = float(input("🔢 Tỉ lệ chướng ngại vật (0-1, mặc định 0.2): ") or 0.2)
                num_targets = int(input("🔢 Số lượng targets (mặc định 1): ") or 1)
                editor.randomize(obstacle_prob=obstacle_prob, num_targets=num_targets, env_type=env_type)
                map_data = {
                    "grid": editor.grid,
                    "start": editor.start,
                    "goal": editor.goal,
                    "targets": editor.targets
                }
                np.save(map_file, map_data)
                print(f"💾 Đã lưu bản đồ ngẫu nhiên vào file '{map_file}'")
            except Exception as e:
                print(f"❌ Lỗi khi random: {e}. Sẽ tạo bản đồ trống.")
        editor.run()
        run_algorithms(map_file)
else:
    print("⚠️ Không có bản đồ nào, tạo mới...")
    map_file = "map1.npy"
    w = int(input("🔧 Nhập width bản đồ: "))
    h = int(input("🔧 Nhập height bản đồ: "))
    random_choice = input("🌱 Sinh ngẫu nhiên bản đồ? (Y/n): ").strip().lower()
    editor = InteractiveMapEditor(width=w, height=h, map_file=map_file)
    if random_choice == '' or random_choice == 'y':
        try:
            print("\n🌍 Chọn loại môi trường:")
            print("1. Mặc định (ngẫu nhiên)")
            print("2. Nhà kho (warehouse)")
            print("3. Thành phố (city)")
            print("4. Mê cung (maze)")
            print("5. Rừng (forest)")
            print("6. Núi (mountain)")
            env_choice = input("🔢 Chọn loại môi trường (1-6, mặc định 1): ").strip()
            env_types = {
                '1': 'default',
                '2': 'warehouse',
                '3': 'city',
                '4': 'maze',
                '5': 'forest',
                '6': 'mountain'
            }
            env_type = env_types.get(env_choice, 'default')
            
            obstacle_prob = float(input("🔢 Tỉ lệ chướng ngại vật (0-1, mặc định 0.2): ") or 0.2)
            num_targets = int(input("🔢 Số lượng targets (mặc định 1): ") or 1)
            editor.randomize(obstacle_prob=obstacle_prob, num_targets=num_targets, env_type=env_type)
            map_data = {
                "grid": editor.grid,
                "start": editor.start,
                "goal": editor.goal,
                "targets": editor.targets
            }
            np.save(map_file, map_data)
            print(f"💾 Đã lưu bản đồ ngẫu nhiên vào file '{map_file}'")
        except Exception as e:
            print(f"❌ Lỗi khi random: {e}. Sẽ tạo bản đồ trống.")
    editor.run()
    run_algorithms(map_file)


