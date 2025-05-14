import heapq
import math
import time

import numpy as np

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_neighbors(node, grid):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbors = []
    x, y = node
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            if abs(dx) == 1 and abs(dy) == 1:
                if grid[x + dx][y] == 1 or grid[x][y + dy] == 1:
                    continue
            neighbors.append((nx, ny))
    return neighbors

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

def astar(grid, start, goal, return_visited=False):
    start_time = time.time()

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited_nodes = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()

            if return_visited:
                return path, visited_nodes
            return path

        visited_nodes.add(current)

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + euclidean(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + euclidean(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    if return_visited:
        return [], visited_nodes
    return []


def astar_with_targets(grid, start, targets, goal, return_visited=False):
    full_path = []
    visited_all = set()
    current = start
    points = targets + [goal]

    for point in points:
        path, visited = astar(grid, current, point, return_visited=True)
        if path is None:
            return (None, visited_all) if return_visited else None
        if full_path:
            full_path += path[1:]
        else:
            full_path += path
        visited_all.update(visited)
        current = point

    return (full_path, visited_all) if return_visited else full_path

