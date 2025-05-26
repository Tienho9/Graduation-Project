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

def get_direction_priority(current, goal):
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    directions = []
    if dx < 0: directions.append((-1, 0))
    elif dx > 0: directions.append((1, 0))
    if dy < 0: directions.append((0, -1))
    elif dy > 0: directions.append((0, 1))
    if dx != 0 and dy != 0:
        directions.insert(0, (dx // abs(dx), dy // abs(dy)))
    return directions

def get_neighbors(current, grid, goal=None):
    x, y = current
    preferred = get_direction_priority(current, goal) if goal else []
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
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    x, y = x1, y1
    n = 1 + dx + dy
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    while True:
        if not is_safe(grid, (x, y)):
            return False
        if x != x1 and y != y1:
            if not (is_safe(grid, (x, y1)) and is_safe(grid, (x1, y))):
                return False
        if (x, y) == (x2, y2): break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
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

def astar_improved_with_targets(grid, start, targets, goal, smooth=True):
    full_path = []
    visited_all = set()
    current = start
    points = targets + [goal]
    target_order = []
    for idx, point in enumerate(points):
        sub_path, visited = astar_improved(grid, current, point)
        if sub_path is None:
            return None, visited_all, target_order
        if smooth:
            sub_path = smooth_path(sub_path, grid)
        if full_path:
            full_path += sub_path[1:]
        else:
            full_path += sub_path
        visited_all.update(visited)
        if idx < len(targets):
            for t_idx, t in enumerate(targets):
                if point == t:
                    target_order.append(t_idx)
                    break
        current = point
    return full_path, visited_all, target_order
