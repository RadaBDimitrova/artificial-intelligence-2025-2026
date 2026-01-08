# Defense Documentation: Frog Leap (frog-leap.py)

## Problem Overview
**Task**: Solve the frog leap puzzle where N frogs on the left ('>') need to swap positions with N frogs on the right ('<') using a single empty space ('_').

**Algorithm**: Depth-First Search (DFS) with backtracking

---

## Code Structure Explanation

### 1. State Representation (Lines 1-3)
```python
r = -1  # '>'
l = 1   # '<'
m = 0   # '_'
```
**What it does**: Constants to represent right-facing frogs, left-facing frogs, and empty space.
**Why**: Using integers instead of characters makes comparison and manipulation faster.

### 2. Main DFS Function (Lines 5-35)
```python
def dfs(state, goal_state, solution, empty):
```
**Parameters**:
- `state`: Current board configuration (list)
- `goal_state`: Target configuration
- `solution`: Path of states leading to solution
- `empty`: Index of empty space

**Location**: Lines 5-35

### 3. Base Cases (Lines 6-9)
```python
solution.append(state[:])
if state == goal_state:
    print('\n'.join(''.join('>' if x == r else '<' if x == l else '_' for x in s) for s in solution))
    return True
```
**What it does**: Adds current state to solution path; if goal reached, prints solution.
**Why**: DFS needs to track the path and recognize success.

### 4. Move Generation (Lines 11-35)

#### Move 1: Right frog moves left (Lines 11-15)
```python
if empty >= 1 and state[empty - 1] == r:
```
**Where**: A '>' frog immediately to the left of empty space slides into it.
**Condition**: Empty space not at left edge and left neighbor is '>'.

#### Move 2: Right frog jumps left (Lines 17-21)
```python
if empty >= 2 and state[empty - 2] == r and state[empty - 1] == l:
```
**Where**: A '>' frog two positions left jumps over a '<' frog.
**Condition**: At least 2 spaces to the left, pattern is ">_<".

#### Move 3: Left frog moves right (Lines 23-27)
```python
if empty < len(state) - 1 and state[empty + 1] == l:
```
**Where**: A '<' frog immediately to the right slides left into empty space.
**Condition**: Empty space not at right edge and right neighbor is '<'.

#### Move 4: Left frog jumps right (Lines 29-33)
```python
if empty < len(state) - 2 and state[empty + 1] == r and state[empty + 2] == l:
```
**Where**: A '<' frog two positions right jumps over a '>' frog.
**Condition**: At least 2 spaces to the right, pattern is ">_<".

### 5. Backtracking (Lines 35)
```python
solution.pop()
return False
```
**What it does**: If no move leads to solution, remove current state and backtrack.
**Why**: Essential for DFS to explore alternative paths when stuck.

### 6. Main Function (Lines 38-43)
```python
def main():
    N = int(input())
    start = [r] * N + [m] + [l] * N
    goal_state = [l] * N + [m] + [r] * N
```
**What it does**: 
- Reads N (number of frogs per side)
- Creates initial state: `>>> _ <<<`
- Creates goal state: `<<< _ >>>`
- Calls DFS

**Where solution starts**: Line 42 calls `dfs(start, goal_state, solution, N)`

---

## Implementation Questions & Answers

### Q1: Where exactly does the DFS search begin?
**A**: Line 42 in `main()`. The call is `dfs(start, goal_state, solution, N)` where N is the index of the empty space in the initial configuration.

### Q2: How is backtracking implemented?
**A**: Lines 35 (`solution.pop()`) and the swap reversals after each recursive call (lines 15, 21, 27, 33). After trying a move, if it doesn't lead to solution, we:
1. Reverse the swap to restore state
2. Try next move
3. If all moves fail, pop the current state from solution

### Q3: What prevents infinite loops/cycles?
**A**: The solution list acts as implicit visited tracking. Since we only append states that are progressively closer to the goal and backtrack when stuck, and the problem has a finite solution, we don't revisit states. However, this implementation could be improved with explicit visited set for efficiency.

### Q4: Why do we use `state[:]` in line 6?
**A**: `state[:]` creates a shallow copy. Without it, we'd append references to the same list object, and all entries in `solution` would reflect the final state. The copy preserves each state as it was when added.

### Q5: How many moves are required for N frogs?
**A**: $N^2 + 2N$ moves. For N=3: 15 moves.

### Q6: What is the time complexity?
**A**: 
- **Worst case**: $O(b^d)$ where b is branching factor (~4 moves per state) and d is depth ($N^2 + 2N$)
- **Space**: $O(d)$ for recursion stack
- In practice, much faster due to problem structure

### Q7: Is this search algorithm complete and optimal?
**A**: 
- **Complete**: Yes, DFS will find a solution if one exists (and one always exists for this puzzle)
- **Optimal**: No, DFS finds *a* solution but not necessarily the shortest one. However, for this specific puzzle, the solution path is essentially unique.

---

## Theoretical Questions & Answers

### Q1: What is Depth-First Search?
**A**: DFS is an uninformed search algorithm that explores as deeply as possible along each branch before backtracking. It uses a stack (implicit via recursion here) to track the frontier.

### Q2: What are alternatives to DFS for this problem?
**A**: 
1. **Breadth-First Search (BFS)**: Would guarantee optimal solution, but uses more memory
2. **Iterative Deepening DFS (IDDFS)**: Combines DFS space efficiency with BFS optimality
3. **A* search**: With appropriate heuristic, would be faster and optimal

**Comparison**:
- DFS: O(d) space, not optimal, can be fast
- BFS: O(b^d) space, optimal, slower for deep solutions
- A*: O(b^d) space, optimal with admissible heuristic, fastest with good heuristic

### Q3: Why is this problem solvable?
**A**: The frog leap puzzle is solvable for any N > 0 because:
1. It's a well-studied problem with known solution patterns
2. The state space is connected - you can reach any valid state from any other
3. The goal state is reachable through legal moves

### Q4: How would you add cycle detection?
**A**: Add a `visited` set:
```python
def dfs(state, goal_state, solution, empty, visited):
    state_tuple = tuple(state)
    if state_tuple in visited:
        return False
    visited.add(state_tuple)
    # ... rest of code
```

### Q5: What is the branching factor?
**A**: Maximum 4 moves per state (slide/jump in each direction). In practice, ~2-3 on average due to edge constraints and frog positions.

### Q6: Could you use memoization here?
**A**: Not effectively. Memoization works when subproblems repeat, but in this DFS path, each state configuration appears at most once. The overhead of storing states would exceed benefits.

### Q7: What makes this a search problem?
**A**: 
- **Initial state**: Starting configuration
- **Actions**: Four types of moves
- **Transition model**: How moves change state
- **Goal test**: Does current state match goal?
- **Path cost**: Number of moves (not optimized here)

---

## Quick Reference: Code Locations

| Functionality | Line Numbers |
|---------------|--------------|
| State constants | 1-3 |
| DFS function signature | 5 |
| Solution tracking | 6 |
| Goal check | 8-9 |
| Right frog slides left | 11-15 |
| Right frog jumps left | 17-21 |
| Left frog slides right | 23-27 |
| Left frog jumps right | 29-33 |
| Backtracking | 35 |
| Main initialization | 38-42 |

---

## Common Pitfalls & Edge Cases

1. **N=0**: Empty puzzle, trivially solved
2. **N=1**: One frog each side - solution exists
3. **Large N**: DFS may take time but will find solution
4. **State copying**: Must use `state[:]` not just `state`

---

## Performance Characteristics

- **Time**: Exponential in worst case, but pruned by problem structure
- **Space**: Linear in solution depth due to recursion
- **Optimality**: Not guaranteed (DFS finds *a* solution)
- **Completeness**: Yes (solution always exists)
