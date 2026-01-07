import math
import os
import time

def manhattan_distance(state, goal_pos, size):
    dist = 0
    for idx, val in enumerate(state):
        if val != 0:
            r, c = idx // size, idx % size
            goal_r, goal_c = goal_pos[val]
            dist += abs(r - goal_r) + abs(c - goal_c)
    return dist


def possible_moves(idx, size):
    moves = []
    r, c = idx // size, idx % size
    if r < size - 1:
        moves.append(('up', idx + size))
    if r > 0:
        moves.append(('down', idx - size))
    if c < size - 1:
        moves.append(('left', idx + 1))
    if c > 0:
        moves.append(('right', idx - 1))
    return moves


def swap(board, i, j):
    b = list(board)
    b[i], b[j] = b[j], b[i]
    return tuple(b)


def is_solvable(board, size, blank_row):
    arr = [x for x in board if x != 0]
    inversions = sum(
        1 for i in range(len(arr)) for j in range(i + 1, len(arr)) if arr[i] > arr[j]
    )
    if size % 2 == 1:
        return inversions % 2 == 0
    else:
        if blank_row % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0


def search(state, g, prev_move, threshold, blank_idx, goal_state, size, goal_pos, path, visited):
    f = g + manhattan_distance(state, goal_pos, size)
    if f > threshold:
        return f
    if state == goal_state:
        return True

    minimum = math.inf
    visited.add(state)

    for move, new_blank in possible_moves(blank_idx, size):
        # optimization to avoid reversing the previous move (from link)
        if (prev_move == 'up' and move == 'down') or (prev_move == 'down' and move == 'up') or (prev_move == 'left' and move == 'right') or (prev_move == 'right' and move == 'left'):
            continue

        new_state = swap(state, blank_idx, new_blank)
        if new_state in visited:
            continue

        path.append(move)
        res = search(new_state, g + 1, move, threshold, new_blank, goal_state, size, goal_pos, path, visited) #deepening
        if res is True:
            return True
        if res < minimum:
            minimum = res
        path.pop()
        visited.discard(new_state)

    return minimum


def ida_star(state, goal_state, size, goal_pos):
    threshold = manhattan_distance(state, goal_pos, size)
    visited = set()
    path = []
    
    while True:
        res = search(state, 0, None, threshold, state.index(0), goal_state, size, goal_pos, path, visited)
        if res is True:
            return path
        if res == math.inf:
            return None
        threshold = res


def main():
    N = int(input())
    I = int(input())
    size = int(math.sqrt(N + 1))

    start_state = []
    for _ in range(size):
        start_state.extend(map(int, input().strip().split()))
    start_state = tuple(start_state)

    
    goal_state = list(range(1, N + 1)) + [0]
    if I != -1:
        # since by default blank/zero is at N
        goal_state[I], goal_state[N] = goal_state[N], goal_state[I]
    goal_state = tuple(goal_state)

    goal_pos = {val: (idx // size, idx % size) for idx, val in enumerate(goal_state)}

    blank_row_check = size - (N // size)
    if not is_solvable(start_state, size, blank_row_check):
        print(-1)
        return

    if os.environ.get('FMI_TIME_ONLY', '0') == '1':
        start = time.perf_counter()
        
    path = ida_star(start_state, goal_state, size, goal_pos)
    
    if os.environ.get('FMI_TIME_ONLY', '0') == '1':
        end = time.perf_counter()
        elapsed_ms = int((end - start) * 1000)
        print(f"# TIMES_MS: alg={elapsed_ms}")
        return

    if path is None:
        print(-1)
    else:
        print(len(path))
        for move in path:
            print(move)


main()
