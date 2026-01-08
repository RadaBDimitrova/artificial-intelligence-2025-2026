# Defense Documentation: N-Puzzle with IDA* (npuzzle.py)

## Problem Overview
**Task**: Solve N-puzzle (8-puzzle, 15-puzzle, etc.) using IDA* (Iterative Deepening A*).

**Algorithm**: IDA* with Manhattan distance heuristic.

---

## Code Structure & Key Concepts

### 1. Manhattan Distance Heuristic (Lines 6-13)
```python
def manhattan_distance(state, goal_pos, size):
    dist = 0
    for idx, val in enumerate(state):
        if val != 0:
            r, c = idx // size, idx % size
            goal_r, goal_c = goal_pos[val]
            dist += abs(r - goal_r) + abs(c - goal_c)
    return dist
```

**Formula**: $h(n) = \sum |current_x - goal_x| + |current_y - goal_y|$

**Properties**:
- **Admissible**: Never overestimates (each tile needs at least Manhattan distance moves)
- **Consistent**: $h(n) \leq c(n, n') + h(n')$

**Where used**: Line 50 (f-value calculation), line 117 (initial threshold).

### 2. Move Generation (Lines 16-24)
```python
def possible_moves(idx, size):
    moves = []
    r, c = idx // size, idx % size
    if r < size - 1: moves.append(('up', idx + size))
    if r > 0: moves.append(('down', idx - size))
    if c < size - 1: moves.append(('left', idx + 1))
    if c > 0: moves.append(('right', idx - 1))
    return moves
```

**Naming convention**: Move names describe where blank goes, not tile.
- 'up': Blank moves up (tile below slides up into blank)

**Where**: Line 62 generates moves for current state.

### 3. Solvability Check (Lines 33-44)
```python
def is_solvable(board, size, blank_row):
    arr = [x for x in board if x != 0]
    inversions = sum(1 for i in range(len(arr)) 
                     for j in range(i + 1, len(arr)) 
                     if arr[i] > arr[j])
    
    if size % 2 == 1:
        return inversions % 2 == 0
    else:
        if blank_row % 2 == 0:
            return inversions % 2 == 1
        else:
            return inversions % 2 == 0
```

**Solvability rules**:
- **Odd size** (e.g., 3×3): Solvable if inversions are even
- **Even size** (e.g., 4×4): Depends on blank row and inversions parity

**Inversion**: Pair (a, b) where a appears before b but a > b.

**Where**: Line 135 checks before search.

### 4. IDA* Search (Lines 47-74)

#### Core Search Function
```python
def search(state, g, prev_move, threshold, blank_idx, goal_state, size, goal_pos, path, visited):
    f = g + manhattan_distance(state, goal_pos, size)
    if f > threshold:
        return f  # Pruned: exceeds threshold
    if state == goal_state:
        return True  # Found solution
    
    minimum = math.inf
    visited.add(state)
    
    for move, new_blank in possible_moves(blank_idx, size):
        # Avoid reversing previous move
        if (prev_move == 'up' and move == 'down') or ...:
            continue
        
        new_state = swap(state, blank_idx, new_blank)
        if new_state in visited:
            continue
        
        path.append(move)
        res = search(new_state, g + 1, move, threshold, new_blank, ...)
        if res is True:
            return True
        if res < minimum:
            minimum = res
        path.pop()
        visited.discard(new_state)
    
    return minimum
```

**Key points**:
- **Line 50**: Calculate f = g + h (actual cost + heuristic)
- **Line 51-52**: Prune if f exceeds threshold
- **Line 53-54**: Success check
- **Lines 61-63**: Optimization - don't reverse last move
- **Line 72**: Return minimum f-value exceeding threshold

#### Main IDA* Loop (Lines 77-88)
```python
def ida_star(state, goal_state, size, goal_pos):
    threshold = manhattan_distance(state, goal_pos, size)
    visited = set()
    path = []
    
    while True:
        res = search(state, 0, None, threshold, ...)
        if res is True:
            return path  # Solution found
        if res == math.inf:
            return None  # No solution
        threshold = res  # Increase threshold to minimum f that was pruned
```

**IDA* algorithm**:
1. Start with threshold = h(start)
2. Do depth-first search, pruning when f > threshold
3. If no solution, increase threshold to minimum pruned f-value
4. Repeat until solution found

**Memory**: O(depth) - only stores current path, unlike A* which stores frontier.

### 5. Main Function (Lines 91-128)
```python
def main():
    N = int(input())  # Number of tiles (excluding blank)
    I = int(input())  # Goal blank position index
    size = int(math.sqrt(N + 1))
    
    start_state = []
    for _ in range(size):
        start_state.extend(map(int, input().strip().split()))
    start_state = tuple(start_state)
    
    goal_state = list(range(1, N + 1)) + [0]
    if I != -1:
        goal_state[I], goal_state[N] = goal_state[N], goal_state[I]
    goal_state = tuple(goal_state)
    
    goal_pos = {val: (idx // size, idx % size) 
                for idx, val in enumerate(goal_state)}
```

**Input format**:
- N: Number of tiles (8 for 8-puzzle, 15 for 15-puzzle)
- I: Goal blank position (-1 for bottom-right)
- Next size×size lines: Current board state

**Goal state**: By default 1,2,3,...,N,0 but blank can be repositioned.

---

## Implementation Questions & Answers

### Q1: Where is the heuristic function?
**A**: `manhattan_distance()` (lines 6-13). Calculates sum of Manhattan distances for all tiles.

### Q2: What makes this IDA* and not just IDA?
**A**: **Line 50**: Uses f = g + h (cost + heuristic), not just depth. This is A* evaluation function applied with iterative deepening.

### Q3: Where is the threshold updated?
**A**: **Line 88**: `threshold = res`. The search returns minimum f-value that exceeded threshold, which becomes next threshold.

### Q4: How does it avoid revisiting states?
**A**: 
- **Line 57**: `visited.add(state)` marks state as visited
- **Line 68**: `if new_state in visited: continue` skips visited states
- **Line 73**: `visited.discard(new_state)` removes when backtracking

### Q5: What is the optimization on lines 61-63?
**A**: Prevents reversing the previous move. If last move was 'up', don't move 'down' immediately (returns to previous state).

### Q6: Why return `math.inf` on line 85?
**A**: If search exhausts all possibilities without finding solution and minimum never updated, returns infinity to signal unsolvable.

### Q7: Where is solvability checked?
**A**: **Lines 135-137**. Must check before searching to avoid wasting time on unsolvable puzzles.

### Q8: How is the path tracked?
**A**: 
- **Line 69**: `path.append(move)` adds move
- **Line 72**: `path.pop()` removes on backtrack
- **Line 82**: Returns complete path when solution found

---

## Theoretical Questions & Answers

### Q1: What is IDA* (Iterative Deepening A*)?
**A**: Combines IDA (iterative deepening) with A* heuristic evaluation.

**Algorithm**:
1. Set threshold = h(start)
2. Do DFS with f-limit = threshold
3. If no solution, increase threshold to minimum pruned f-value
4. Repeat

**Benefits**:
- **Memory**: O(depth) like DFS
- **Optimal**: Yes, if heuristic admissible
- **Complete**: Yes, if solution exists

### Q2: Compare IDA* to A* and IDA.

| Aspect | A* | IDA | IDA* |
|--------|-----|-----|------|
| Memory | O(b^d) | O(d) | O(d) |
| Optimal | Yes (admissible h) | Yes | Yes (admissible h) |
| Time | O(b^d) | O(b^d) | O(b^d) |
| Heuristic | Yes | No | Yes |
| Best for | Small state space | No heuristic | Large state space |

### Q3: What makes Manhattan distance admissible?
**A**: **Never overestimates** true cost.

**Proof**: Each tile must move at least its Manhattan distance. Since tiles block each other, actual cost ≥ Manhattan distance.

**Required for**: IDA* optimality.

### Q4: What makes Manhattan distance consistent?
**A**: **Triangle inequality**: $h(n) \leq c(n, n') + h(n')$

One move changes one tile's position by 1, so Manhattan distance changes by at most 1. Since move cost = 1, consistency holds.

**Benefit**: No need to reopen nodes.

### Q5: Time and space complexity?
**A**: 
**Time**: O(b^d) where b = branching factor (~3 after pruning), d = solution depth.

**Space**: O(d) - only current path stored.

**Practical**: 
- 8-puzzle: Solvable in seconds (d ≤ 31)
- 15-puzzle: Minutes to hours (d ≤ 80)
- 24-puzzle: Impractical

### Q6: What is the branching factor?
**A**: **Theoretical**: 4 moves (up/down/left/right).

**Practical**: ~3 after pruning:
- Edge constraints reduce options
- Don't reverse previous move (line 61-63)
- Visited set prevents cycles (line 68)

### Q7: Why is solvability checking important?
**A**: Half of random configurations are unsolvable!

**Without check**: Algorithm would search forever on unsolvable puzzle.

**Efficiency**: O(n²) check saves potentially infinite search time.

### Q8: Is IDA* complete and optimal?
**A**: 
- **Complete**: Yes, if branching factor finite and solution exists
- **Optimal**: Yes, if heuristic is admissible

**This implementation**: Both properties hold.

### Q9: What are alternative heuristics?
**A**: 
1. **Manhattan distance** (this implementation): Admissible, fast
2. **Linear conflict**: Manhattan + penalties for blocking tiles (better but slower)
3. **Pattern database**: Precomputed optimal costs for subproblems (very powerful)
4. **Misplaced tiles**: Count tiles out of position (weaker than Manhattan)

### Q10: When does IDA* perform poorly?
**A**: 
**Many shallow solutions**: Revisits same states at different thresholds.

**Deep solutions**: Must iterate through many thresholds.

**Example**: If optimal depth is 50, iterates 50 times, redoing shallow work each time.

**Better**: A* with sufficient memory, or bidirectional search.

### Q11: What is iterative deepening?
**A**: Strategy that runs depth-limited DFS with increasing depth limits.

**Process**:
1. DFS with depth limit 1
2. DFS with depth limit 2
3. Continue until solution found

**Overhead**: Revisits shallower levels, but asymptotically negligible.

**IDA***: Same concept but with f-value threshold instead of depth.

### Q12: How does the optimization on line 61-63 help?
**A**: **Reduces branching factor from 4 to 3** (25% reduction).

**Without**: Up → Down → Up → Down ... infinite cycle (prevented by visited set but wastes checks).

**With**: Immediate reversal impossible, search more efficient.

**Impact**: Significant on deep searches (3^d vs 4^d).

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Manhattan distance | 6-13 |
| Move generation | 16-24 |
| State swap | 27-30 |
| Solvability check | 33-44 |
| IDA* search core | 47-74 |
| IDA* main loop | 77-88 |
| Main function | 91-128 |

---

## Performance Characteristics

**8-puzzle** (3×3):
- States: 9!/2 ≈ 181,440
- Max optimal: 31 moves
- Time: Milliseconds to seconds

**15-puzzle** (4×4):
- States: 16!/2 ≈ 10.5 trillion
- Max optimal: 80 moves
- Time: Seconds to hours (depending on configuration)

**Memory**: O(d) ≈ few KB even for 15-puzzle.

**Optimality**: Guaranteed (Manhattan distance is admissible).
