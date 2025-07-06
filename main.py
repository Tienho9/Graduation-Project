import os
import time
import math
import numpy as np
from tabulate import tabulate
from map_utils import InteractiveMapEditor
from astar_trad import astar_with_targets
from astar_imp import astar_improved_with_targets
from astar_imp_with_greedy import astar_improved_with_targets_greedy
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

def run_algorithms(map_file, selected_algo=None):
    # Hàm này được thiết kế để gọi từ GUI hoặc dòng lệnh
    # Trả về kết quả mà không in hoặc hiển thị trực tiếp

    data = np.load(map_file, allow_pickle=True).item()
    grid = data.get("grid")
    start = data.get("start")
    goal = data.get("goal")
    targets = data.get("targets", [])

    if isinstance(start, np.ndarray):
        start = tuple(start.tolist())
    if isinstance(goal, np.ndarray):
        goal = tuple(goal.tolist())
    targets = [tuple(t.tolist()) if isinstance(t, np.ndarray) else t for t in targets]

    if start is None or goal is None:
        print("❌ Thiếu Start hoặc Goal!") 
        return None, None, None, None

    
    algo_info = {
        "A*": (astar_with_targets, "▶️ A*...", True),
        "ImpA*": (astar_improved_with_targets, "✨ ImpA*...", False),
        "ImpA*G": (astar_improved_with_targets_greedy, "💡 ImpA*G...", False),
        "ImpA*ACO": (astar_improved_with_targets_aco, "🌟 ImpA*ACO...", False)
    }

    paths = {}
    visited_nodes_raw = {}
    orders = {}
    metrics = {}

    algos_to_run = [selected_algo] if selected_algo and selected_algo != "All Algorithms" else algo_info.keys()
    
    for algo_name in algos_to_run:
        if algo_name in algo_info:
            algo_func, print_msg, supports_return_visited = algo_info[algo_name]
            t0 = time.time()

            if supports_return_visited:
                result = algo_func(grid, start, targets, goal, return_visited=True)
            else:
                result = algo_func(grid, start, targets, goal)

            runtime = time.time() - t0

            if result and isinstance(result, tuple) and len(result) >= 2:
                 path = result[0] if len(result) > 0 else None
                 visited = result[1] if len(result) > 1 else set()
                 order = result[2] if len(result) > 2 else []
            else:
                path, visited, order = None, set(), []
                print(f"❌ {algo_name} thất bại hoặc trả về sai định dạng!") # Keep print for command line feedback

            planning_time = runtime 
            stat = compute_path_stats(path, visited, runtime, planning_time)

            paths[algo_name] = path
            visited_nodes_raw[algo_name] = visited
            orders[algo_name] = order
            metrics[algo_name] = {
                "Runtime (s)": stat[0] if stat is not None and len(stat) > 0 else None,
                "Planning (s)": stat[1] if stat is not None and len(stat) > 1 else None,
                "Nodes": stat[2] if stat is not None and len(stat) > 2 else None,
                "Inflect.": stat[3] if stat is not None and len(stat) > 3 else None,
                "Angle": stat[4] if stat is not None and len(stat) > 4 else None,
                "Length": stat[5] if stat is not None and len(stat) > 5 else None,
                "Target Order": order 
            }
            print(f"➡️ {algo_name} trả về thứ tự: {order}") 
        else:
            print(f"⚠️ Thuật toán {algo_name} không tìm thấy hoặc không được hỗ trợ.") # Keep print for command line feedback

    return paths, visited_nodes_raw, orders, metrics

def calculate_overall_scores(metrics_data):
  
    weights = {'Length': 0.4, 'Inflect.': 0.3, 'Angle': 0.3}
    keys = list(weights.keys())
    scores = {}
    for algo, m in metrics_data.items():
        score = 0.0
        for k in keys:
            v = m.get(k, 0)
            score += weights[k] * v
        scores[algo] = score
    return scores

def main():
   
    saved_maps = [f for f in os.listdir() if f.endswith(".npy")]
    if saved_maps:
        print("\n📂 Danh sách bản đồ đã lưu:")
        for i, f in enumerate(saved_maps):
            print(f"{i + 1}. {f}")
        choice = input("🔢 Chọn bản đồ (số) hoặc nhập tên mới: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(saved_maps):
            map_file = saved_maps[int(choice) - 1]
            try:
                map_data = np.load(map_file, allow_pickle=True).item()
                if not all(k in map_data for k in ["grid", "start", "goal", "targets"]):
                    print(f"❌ File '{map_file}' thiếu trường dữ liệu cần thiết (grid, start, goal, targets). Hãy tạo lại bản đồ!")
                    return 
            except FileNotFoundError:
                print(f"❌ Không tìm thấy file '{map_file}'. Hãy kiểm tra lại tên file!")
                return 
            except Exception as e:
                print(f"❌ Lỗi khi đọc file '{map_file}': {e}. Hãy kiểm tra lại file!")
                return 

            
            paths, visited_nodes_raw, orders, metrics = run_algorithms(map_file)
            
            
            print("\n📊 SO SÁNH THUẬT TOÁN")
            headers = ["Info", "A*", "ImpA*", "ImpA*G", "ImpA*ACO"]
            
            metric_rows = []
            
            if metrics:
                 sample_metrics = list(metrics.values())[0]
                 metric_names = [key for key in sample_metrics.keys() if key != "Target Order"]
                 for metric_name in metric_names:
                      row = [metric_name] + [metrics.get(algo, {}).get(metric_name, "-") for algo in algo_info.keys()]
                      formatted_row = [row[0]] + [safe_format(val, "{:.3f}") if isinstance(val, float) and metric_name in ["Runtime (s)", "Planning (s)"] else safe_format(val, "{:.1f}") if isinstance(val, float) and metric_name in ["Angle", "Length"] else str(val) for val in row[1:]]
                      metric_rows.append(formatted_row)
                      
                 
                 order_row = ["Target Order"] + [' → '.join([str(t_idx + 1) for t_idx in orders.get(algo, [])]) if orders.get(algo) else "-" for algo in algo_info.keys()]
                 metric_rows.append(order_row)

                 print(tabulate(metric_rows, headers=headers, tablefmt="fancy_grid"))

            print("\n🔍 CHI TIẾT ĐƯỜNG ĐI:")
            
            for algo_name in algo_info.keys():
                path = paths.get(algo_name)
                if path:
                    print(f"\n{algo_name}:")
                    print("Đường đi:", path)

           
            import matplotlib.pyplot as plt
            print("🎨 Nhấn phím [1-5] để hiện/ẩn các đường đi:")
            
            paths_list = [paths.get(algo, []) for algo in algo_info.keys()]
            orders_list = [orders.get(algo, []) for algo in algo_info.keys()]

            fig = plot_grid_map_compare(
                grid, start, goal,
                *paths_list, 
                targets,
                targets_ordered_list=orders_list 
            )
      
            def order_text_cmd(order_list, targets_list):
                if not order_list or not targets_list:
                    return "Không có target nào."
                return ' → '.join([f"T{t_idx + 1}" for t_idx in order_list])

            y_pos = 0.10  
            colors = ['red', 'orange', 'blue', 'purple', 'green'] 
            for i, algo_name in enumerate(algo_info.keys()):
                 order = orders.get(algo_name, [])
                 formatted_order_text = order_text_cmd(order, targets)
                 plt.figtext(0.05, y_pos, f"{algo_name}: {formatted_order_text}", 
                           ha='left', fontsize=8, color=colors[i % len(colors)])
                 y_pos -= 0.02 

            sx, sy = start
            ax.add_patch(plt.Rectangle((sy, height - sx - 1), 1, 1, color='green', label='Start'))
            gx, gy = goal
            ax.add_patch(plt.Rectangle((gy, height - gx - 1), 1, 1, color='red', label='Goal'))
            if targets:
                for idx, (tx, ty) in enumerate(targets):
                    ax.add_patch(plt.Rectangle((ty, height - tx - 1), 1, 1, color='orange', label='Target' if idx == 0 else ""))
                    ax.text(ty + 0.5, height - tx - 0.5, f"T{idx+1}", color='blue', fontsize=7, ha='center', va='center', fontweight='bold')

            plt.show()

        else:
       
            map_file = f"{choice}.npy"
            try:
                w = int(input("🔧 Nhập width bản đồ: "))
                h = int(input("🔧 height bản đồ: "))
                random_choice = input("🌱 Sinh ngẫu nhiên bản đồ? (Y/n): ").strip().lower()
                editor = InteractiveMapEditor(width=w, height=h, map_file=map_file)

                if random_choice == '' or random_choice == 'y':
                    print("\n🌍 Chọn loại môi trường:")
                    print("1. Mặc định (ngẫu nhiên)")
                    print("2. Nhà kho (warehouse)")
                    print("3. Thành phố (city)")
                    env_choice = input("🔢 Chọn loại môi trường (1-3, mặc định 1): ").strip()
                    env_types = {
                        '1': 'default',
                        '2': 'warehouse',
                        '3': 'city'
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
                else:
                    print("🗺️ Tạo bản đồ trống.")
                
               
                run_algorithms(map_file)
                
                
            except ValueError:
                print("❌ Đầu vào không hợp lệ. Vui lòng nhập số.")
                return
            except Exception as e:
                print(f"❌ Lỗi khi tạo bản đồ: {e}")
                return
            
          

    else:
        
        print("⚠️ Không có bản đồ nào, tạo mới...")
        map_file = "map1.npy"
        try:
            w = int(input("🔧 Nhập width bản đồ: "))
            h = int(input("🔧 height bản đồ: "))
            random_choice = input("🌱 Sinh ngẫu nhiên bản đồ? (Y/n): ").strip().lower()
            editor = InteractiveMapEditor(width=w, height=h, map_file=map_file)

            if random_choice == '' or random_choice == 'y':
                 print("\n🌍 Chọn loại môi trường:")
                 print("1. Mặc định (ngẫu nhiên)")
                 print("2. Nhà kho (warehouse)")
                 print("3. Thành phố (city)")
                 env_choice = input("🔢 Chọn loại môi trường (1-3, mặc định 1): ").strip()
                 env_types = {
                     '1': 'default',
                     '2': 'warehouse',
                     '3': 'city'
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
            else:
                print("🗺️ Tạo bản đồ trống.")

            
            run_algorithms(map_file)

            

        except ValueError:
             print("❌ Đầu vào không hợp lệ. Vui lòng nhập số.")
             return
        except Exception as e:
             print(f"❌ Lỗi khi tạo bản đồ: {e}")
             return


if __name__ == '__main__':
    main()


