import heapq
import math
import numpy as np
from astar_imp import astar_improved, smooth_path

def _calculate_path_length(path):
    
    return sum(math.hypot(path[i][0] - path[i+1][0], path[i][1] - path[i+1][1]) for i in range(len(path)-1)) if path and len(path) > 1 else 0

def astar_improved_with_targets_greedy(grid, start, targets, goal):
    
    full_path = []
    visited_all = set()
    current = start
    remaining_targets = list(targets)
    target_order = []

    while remaining_targets:
        best_next_target = None
        min_path_cost = float('inf')
        path_to_best_target = None
        visited_for_best_path = set()

        for candidate_target in remaining_targets:
            path, visited = astar_improved(grid, current, candidate_target)
            
            if path:
                current_path_cost = _calculate_path_length(path)
                if current_path_cost < min_path_cost:
                    min_path_cost = current_path_cost
                    best_next_target = candidate_target
                    path_to_best_target = path
                    visited_for_best_path = visited
        
        if best_next_target:
            if full_path:
                full_path.extend(path_to_best_target[1:])
            else:
                full_path.extend(path_to_best_target)

            visited_all.update(visited_for_best_path)
            target_order.append(targets.index(best_next_target))
            
            current = best_next_target
            remaining_targets.remove(best_next_target)
        else:
            return None, visited_all, target_order

    final_leg_path, final_leg_visited = astar_improved(grid, current, goal)
    if final_leg_path:
        if full_path:
            full_path.extend(final_leg_path[1:])
        else:
            full_path.extend(final_leg_path)
        visited_all.update(final_leg_visited)
    else:
        return None, visited_all, target_order

    # Thêm bước làm mượt cuối cùng
    must_include_points = {start, goal} | set(targets)
    smoothed_path = smooth_path(full_path, grid, must_include=must_include_points)
    
    return smoothed_path, visited_all, target_order
