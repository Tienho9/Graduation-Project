import heapq
import math
import numpy as np
import random

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

    # Tính góc giữa hướng đi hiện tại và trục X
    
    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

    # Định nghĩa 8 hướng đi
    all_directions = {
        'N': (-1, 0), 'NE': (-1, 1), 'E': (0, 1), 'SE': (1, 1),
        'S': (1, 0), 'SW': (1, -1), 'W': (0, -1), 'NW': (-1, -1)
    }

    # Bảng quy tắc chọn 3 hướng theo từng góc 
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

    # Bảng quy tắc chọn 5 hướng 
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
        direction_set = direction_groups_3  
    elif obstacle_count < 4:
        direction_set = direction_groups_5  
    else:
        return list(all_directions.values())  

    # Chọn hướng phù hợp theo góc angle
    for (start, end), names in direction_set.items():
        if start <= angle < end or (start > end and (angle >= start or angle < end)):
            return [all_directions[d] for d in names]

    # Nếu không khớp vùng nào mở toàn bộ hướng 
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

def obstacle_info(current, grid, params=None):
    x, y = current
    rows, cols = len(grid), len(grid[0])
    penalty = 0
    radius = params.get('radius', 3) if params else 3
    penalty_factor = params.get('penalty_factor', 1/20) if params else 1/20
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                dist = max(1, math.hypot(dx, dy))
                penalty += penalty_factor * (1 / dist)
    return penalty

def near_corner_penalty(current, grid):
    x, y = current
    penalty = 0
    for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 1:
            penalty += 0.5
    return penalty

def calculate_weights(current, goal, grid, params=None):
    base_alpha = params.get('base_alpha', 0.5) if params else 0.5
    radius = params.get('radius', 3) if params else 3
    rows, cols = grid.shape
    obstacles = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not obstacles:
        return base_alpha, 1.0 - base_alpha
    near = sum(1 for obs in obstacles if euclidean(current, obs) < radius)
    density = min(near / (np.pi * radius ** 2), 1.0)
    alpha = base_alpha * (1 - density)
    beta = 1.0 - alpha
    return alpha, beta

def evaluation_function(current, goal, start, g_score, grid, params=None):
    G = g_score
    H = euclidean(current, goal)
    R = euclidean(start, goal)
    r = euclidean(current, goal)
    c_factor = params.get('c_factor', 1/50) if params else 1/50
    C = c_factor * ((R - r) / R) * H if R != 0 else 0
    I = obstacle_info(current, grid, params=params) + near_corner_penalty(current, grid)
    alpha, beta = calculate_weights(current, goal, grid, params=params)
    o = -alpha * C + beta * I
    o = np.clip(o, -H, H)
    return G + H + o

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar_improved(grid, start, goal, params=None):
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
            f = evaluation_function(neighbor, goal, start, tentative_g, grid, params=params)
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
            if e2 < dx: 
                if not is_safe(grid, (x + sx, y)) or not is_safe(grid, (x, y + sy)):
                    return False
            err -= dy
            x += sx
        
        if e2 < dx:
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



def aco_tsp(dist_matrix, params=None):
    n_ants = int(params.get('n_ants', 20)) if params else 20
    n_best = 2  
    n_iterations = int(params.get('n_iterations', 50)) if params else 50
    decay = params.get('decay', 0.1) if params else 0.1
    alpha = 1  
    beta = 2  
    n = len(dist_matrix)
    pheromone = np.ones((n, n))
    best_path, best_cost = None, float('inf')
    for _ in range(n_iterations):
        all_paths = []
        for _ in range(n_ants):
            path = []
            visited = set()
            current = 0  
            path.append(current)
            visited.add(current)
            for _ in range(n - 2): 
                probs = []
                for j in range(1, n - 1):
                    if j in visited:
                        probs.append(0)
                    else:
                        tau = pheromone[current][j] ** alpha
                        eta = (1 / dist_matrix[current][j]) ** beta if dist_matrix[current][j] > 0 else 0
                        probs.append(tau * eta)
                probs_sum = sum(probs)
                if probs_sum == 0:
                    break
                probs = [p / probs_sum for p in probs]
                next_node = random.choices(range(1, n - 1), weights=probs, k=1)[0]
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            path.append(n - 1)  
            cost = sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
            all_paths.append((path, cost))
        all_paths.sort(key=lambda x: x[1])
        for path, cost in all_paths[:n_best]:
            for i in range(len(path)-1):
                pheromone[path[i]][path[i+1]] += 1.0 / cost
        pheromone *= (1 - decay)
        if all_paths[0][1] < best_cost:
            best_path, best_cost = all_paths[0]
    return best_path

def compute_distance_matrix(grid, points, pathfinder, params=None):
    n = len(points)
    dist_matrix = np.full((n, n), np.inf)
    paths = {}
    path_cache = {}
    total_visited_nodes = set()

    for i in range(n):
        for j in range(n):
            if i != j:
                cache_key = (points[i], points[j])
                if cache_key in path_cache:
                    path = path_cache[cache_key]
                else:
                    path, visited = pathfinder(grid, points[i], points[j], params=params)
                    total_visited_nodes.update(visited)
                    path_cache[cache_key] = path
                if path:
                    cost = sum(euclidean(path[k], path[k+1]) for k in range(len(path)-1))
                    dist_matrix[i][j] = cost
                    paths[(i, j)] = path
    return dist_matrix, paths, total_visited_nodes


def astar_improved_with_targets_aco(grid, start, targets, goal, smooth=True, params=None):
    points = [start] + targets + [goal]
    dist_matrix, path_lookup, visited_all = compute_distance_matrix(grid, points, astar_improved, params=params)
    
    route_indices = aco_tsp(dist_matrix, params=params)
    if route_indices is None:
        return None, set(), []
        
    full_path = []
    target_order = []
    for i in range(len(route_indices) - 1):
        u, v = route_indices[i], route_indices[i+1]
        path = path_lookup.get((u, v))
        if path is None:
            return None, visited_all, target_order
        if full_path:
            full_path += path[1:]
        else:
            full_path += path
        visited_all.update(path)
        if 1 <= v <= len(targets):
            target_order.append(v - 1)
            
    if smooth:
        must_include_points = [start, goal] + list(targets)
        full_path = smooth_path(full_path, grid, must_include=must_include_points)
        
    return full_path, visited_all, target_order
