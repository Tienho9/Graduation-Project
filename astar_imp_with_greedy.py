import heapq
import math
import numpy as np

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_safe(grid, node):
    r, c = node
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0

def is_safe_diagonal(grid, current, neighbor):
    x, y = current
    nx, ny = neighbor
    dx, dy = nx - x, ny - y
    if not (is_safe(grid, (x + dx, y)) and is_safe(grid, (x, y + dy))):
        return False
    if not is_safe(grid, (x + dx, y + dy)):
        return False
    return True

def get_direction_priority(current, goal, grid, radius=2):
    # Tính độ chênh giữa goal và vị trí hiện tại
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]

    # Tính góc (theo độ) giữa hướng đi hiện tại và trục X
    # Chú ý: trục Y bị đảo vì trong lưới thường y tăng xuống dưới
    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

    # Định nghĩa 8 hướng đi
    all_directions = {
        'N': (-1, 0), 'NE': (-1, 1), 'E': (0, 1), 'SE': (1, 1),
        'S': (1, 0), 'SW': (1, -1), 'W': (0, -1), 'NW': (-1, -1)
    }

    # Bảng quy tắc chọn 3 hướng theo từng góc (trường hợp không có vật cản)
    direction_groups_3 = {
        (337.5, 360): ['NW', 'N', 'NE'], (0, 22.5): ['NW', 'N', 'NE'],
        (22.5, 67.5): ['N', 'NE', 'E'],
        (67.5, 112.5): ['NE', 'E', 'SE'],
        (112.5, 157.5): ['E', 'SE', 'S'],
        (157.5, 202.5): ['SE', 'S', 'SW'],
        (202.5, 247.5): ['S', 'SW', 'W'],
        (247.5, 292.5): ['SW', 'W', 'NW'],
        (292.5, 337.5): ['W', 'NW', 'N'],
    }

    # Bảng quy tắc chọn 5 hướng nếu có ít vật cản
    direction_groups_5 = {
        (337.5, 360): ['N', 'NE', 'E', 'W', 'NW'], (0, 22.5): ['N', 'NE', 'E', 'W', 'NW'],
        (22.5, 67.5): ['N', 'NE', 'E', 'SE', 'NW'],
        (67.5, 112.5): ['NE', 'E', 'SE', 'S', 'N'],
        (112.5, 157.5): ['E', 'SE', 'S', 'SW', 'NE'],
        (157.5, 202.5): ['SE', 'S', 'SW', 'W', 'E'],
        (202.5, 247.5): ['S', 'SW', 'W', 'NW', 'SE'],
        (247.5, 292.5): ['SW', 'W', 'NW', 'N', 'S'],
        (292.5, 337.5): ['W', 'NW', 'N', 'NE', 'SW'],
    }

    # Đếm số vật cản xung quanh trong bán kính nhất định
    def count_obstacles(grid, pos, radius):
        rows, cols = len(grid), len(grid[0])
        x0, y0 = pos
        count = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x0 + dx, y0 + dy
                if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                    count += 1
        return count

    obstacle_count = count_obstacles(grid, current, radius)

    # Chọn chiến lược mở rộng hướng đi dựa trên mật độ vật cản
    if obstacle_count == 0:
        direction_set = direction_groups_3  # An toàn ➜ chỉ mở 3 hướng ưu tiên
    elif obstacle_count < 4:
        direction_set = direction_groups_5  # Có vài vật cản ➜ mở rộng 5 hướng
    else:
        return list(all_directions.values())  # Môi trường phức tạp ➜ mở đủ 8 hướng

    # Chọn hướng phù hợp theo góc angle
    for (start_angle, end_angle), names in direction_set.items():
        if start_angle <= angle < end_angle or (start_angle > end_angle and (angle >= start_angle or angle < end_angle)):
            return [all_directions[d] for d in names]

    # Nếu không khớp vùng nào ➜ mở toàn bộ hướng (fallback)
    return list(all_directions.values())


def get_neighbors(current, grid, goal=None):
    x, y = current
    preferred = get_direction_priority(current, goal, grid) if goal else []

    fallback = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    directions = preferred + [d for d in fallback if d not in preferred]

    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if not is_safe(grid, (nx, ny)):
            continue
        if abs(dx) == 1 and abs(dy) == 1:
            if not is_safe_diagonal(grid, (x, y), (nx, ny)):
                continue
        neighbors.append(((nx, ny), math.hypot(dx, dy)))
    return neighbors

def obstacle_info(current, grid, radius=3):
    x, y = current
    rows, cols = len(grid), len(grid[0])
    penalty = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                dist = max(1, math.hypot(dx, dy))
                penalty += (1 / 20) * (1 / dist)
    return penalty

def near_corner_penalty(current, grid):
    x, y = current
    penalty = 0
    for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
            penalty += 0.5
    return penalty

def calculate_weights(current, goal, grid, base_alpha=0.5, radius=3):
    rows, cols = grid.shape
    obstacles = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not obstacles:
        return 0.5, 0.5
    near = sum(1 for obs in obstacles if euclidean(current, obs) < radius)
    density = min(near / (np.pi * radius ** 2), 1.0)
    alpha = base_alpha * (1 - density)
    beta = 1.0 - alpha
    return alpha, beta

def evaluation_function(current, goal, start, g_score, grid):
    G = g_score
    H = euclidean(current, goal)
    R = euclidean(start, goal)
    r = euclidean(current, goal)
    C = (1 / 50) * ((R - r) / R) * H if R != 0 else 0
    I = obstacle_info(current, grid) + near_corner_penalty(current, grid)
    alpha, beta = calculate_weights(current, goal, grid)
    o = -alpha * C + beta * I
    o = np.clip(o, -H, H)
    return G + H + o

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar_improved(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current), visited
        visited.add(current)
        for neighbor, cost in get_neighbors(current, grid, goal):
            if neighbor in visited:
                continue
            tentative_g = g_score[current] + cost
            f = evaluation_function(neighbor, goal, start, tentative_g, grid)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (f, neighbor))
    return None, visited

def is_line_clear(p1, p2, grid):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)

    x, y = x1, y1
    err = dx - dy

    while True:
        if not is_safe(grid, (x, y)):
            return False
        if (x, y) == (x2, y2):
            break
        
        e2 = 2 * err
        moved_diag = False
        if e2 > -dy:
            # Sắp di chuyển theo trục x
            if e2 < dx: 
                # Nếu cũng sẽ di chuyển theo trục y -> đây là bước đi chéo
                # Trước khi đi chéo, kiểm tra các góc
                if not is_safe(grid, (x + sx, y)) or not is_safe(grid, (x, y + sy)):
                    return False
            err -= dy
            x += sx
        
        if e2 < dx:
            # Di chuyển theo trục y
            err += dx
            y += sy

    return True

def smooth_path(path, grid, must_include=None):
    if len(path) <= 2:
        return path
    must_include = set(must_include) if must_include else set()
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if any(p in must_include for p in path[i+1:j]):
                j -= 1
                continue
            if is_line_clear(path[i], path[j], grid):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed

def astar_improved_with_targets_greedy(grid, start, targets, goal, smooth=True):
    remaining_targets = targets[:]
    current = start
    full_path = []
    visited_all = set()
    target_order = []

    # Xử lý các mục tiêu trung gian
    while remaining_targets:
        nearest = min(remaining_targets, key=lambda t: euclidean(current, t))
        
        for t_idx, t in enumerate(targets):
            if nearest == t:
                target_order.append(t_idx)
                break
        
        path, visited = astar_improved(grid, current, nearest)
        if not path:
            return None, visited_all, target_order
        
        if path and path[0] != current:
            path.insert(0, current)

        
        if not full_path:
            full_path += path  
        else:
            full_path += path[1:]  
        
        visited_all.update(visited)
        current = nearest
        remaining_targets.remove(nearest)

    path, visited = astar_improved(grid, current, goal)
    if path:
        if path and path[0] != current:
            path.insert(0, current)

        if not full_path:    
            full_path += path
        else:
           
            full_path += path[1:]
            
        visited_all.update(visited)

    # Làm mượt đường đi cuối cùng
    if smooth:
        must_include_points = [start, goal] + list(targets)
        full_path = smooth_path(full_path, grid, must_include=must_include_points)

    # Đảm bảo điểm bắt đầu luôn có trong path (an toàn)
    if full_path and full_path[0] != start:
        full_path.insert(0, start)
        
    return full_path, visited_all, target_order
