# Defense Documentation: Tic-Tac-Toe with Minimax (tictactoe.cpp)

## Problem Overview
**Task**: Implement perfect Tic-Tac-Toe AI using Minimax algorithm with alpha-beta pruning.

**Algorithm**: Minimax with alpha-beta pruning for optimal play.

---

## Code Structure & Key Concepts

### 1. Board Representation (Line 5)
```cpp
char board[9] = {'_','_','_','_','_','_','_','_','_'};
```

**1D array**: Position mapping:
```
0 1 2
3 4 5
6 7 8
```

**Helper** (lines 7-9):
```cpp
inline int idx(int r, int c) {
    return (r * 3 + c);
}
```

### 2. Win Detection (Lines 11-20)
```cpp
inline bool isWinner(char p) {
    int wins[8][3] = {
        {0,1,2}, {3,4,5}, {6,7,8},  // Rows
        {0,3,6}, {1,4,7}, {2,5,8},  // Columns
        {0,4,8}, {2,4,6}            // Diagonals
    };
    for (int i = 0; i < 8; i++) {
        if (board[wins[i][0]] == p && 
            board[wins[i][1]] == p && 
            board[wins[i][2]] == p) {
            return true;
        }
    }
    return false;
}
```

**8 winning lines**: 3 rows + 3 columns + 2 diagonals.

### 3. Terminal State Check (Lines 23-37)
```cpp
inline bool isEndState(char& winner) {
    if (isWinner('X')) {
        winner = 'X';
        return true;
    }
    if (isWinner('O')) {
        winner = 'O';
        return true;
    }
    
    for (int i = 0; i < 9; i++) {
        if (board[i] == '_') {
            winner = '_';
            return false;  // Game continues
        }
    }
    
    winner = '_';
    return true;  // Draw
}
```

**Three outcomes**: X wins, O wins, draw.

### 4. Minimax with Alpha-Beta Pruning (Lines 40-88)

```cpp
pair<int, int> minimax(char rootMove, char currentMove, int depth, int alpha, int beta) {
    char winner;
    
    // Terminal states
    if (isEndState(winner)) {
        if (winner == rootMove) {
            return {1000 - depth, -1};  // Win (prefer faster)
        } else if (winner == '_') {
            return {0, -1};              // Draw
        } else {
            return {-1000 + depth, -1};  // Loss (delay if inevitable)
        }
    }
    
    bool maximizing = (currentMove == rootMove);
    int bestScore = maximizing ? INT_MIN : INT_MAX;
    int bestMove = -1;
    char nextPlayer = (currentMove == 'X' ? 'O' : 'X');
    
    for (int i = 0; i < 9; i++) {
        if (board[i] == '_') {
            board[i] = currentMove;
            int sc = minimax(rootMove, nextPlayer, depth + 1, alpha, beta).first;
            board[i] = '_';
            
            if (maximizing) {
                if (sc > bestScore) {
                    bestScore = sc;
                    bestMove = i;
                }
                alpha = max(alpha, bestScore);
            } else {
                if (sc < bestScore) {
                    bestScore = sc;
                    bestMove = i;
                }
                beta = min(beta, bestScore);
            }
            
            if (beta <= alpha) {
                break;  // Alpha-beta pruning
            }
        }
    }
    return {bestScore, bestMove};
}
```

**Key features**:
1. **Lines 44-53**: Terminal state evaluation with depth bonus.
2. **Lines 55-57**: Max or min depending on whose turn.
3. **Lines 62-64**: Try move, recurse, undo.
4. **Lines 66-78**: Update best move and alpha/beta.
5. **Lines 80-82**: Prune when beta ≤ alpha.

**Scoring**:
- Win: 1000 - depth (prefer faster wins)
- Loss: -1000 + depth (delay losses)
- Draw: 0

### 5. Evaluation Function (Lines 91-110)
```cpp
inline int eval(char playerChar) {
    char winner;
    if (isEndState(winner)) {
        return -1;  // Game over, no move
    }
    
    bool isEmpty = true;
    for (int i = 0; i < 9; i++) {
        if (board[i] != '_') {
            isEmpty = false;
            break;
        }
    }
    if (isEmpty) {
        return 4;  // First move: center
    }
    
    auto res = minimax(playerChar, playerChar, 0, INT_MIN / 4, INT_MAX / 4);
    return res.second;  // Return best move index
}
```

**Special cases**:
- **Line 105**: Empty board → play center (position 4).
- **Line 108**: Call minimax for other cases.

### 6. Game Modes

#### Judge Mode (Lines 154-172)
```cpp
if (mode == "JUDGE") {
    string turnLine;
    getline(cin, turnLine);
    char playerChar = turnLine.back();
    
    getBoard(cin);
    char winner;
    if (isEndState(winner)) {
        cout << "-1";
        return 0;
    }
    
    int move = eval(playerChar);
    if (move < 0) {
        cout << "-1";
    } else {
        int rr = (move / 3) + 1;
        int cc = (move % 3) + 1;
        cout << rr << " " << cc << "\n";
    }
}
```

**Purpose**: Given board state, output best move.

#### Game Mode (Lines 174-215)
```cpp
else if (mode == "GAME") {
    string start, player;
    getline(cin, start);
    getline(cin, player);
    char startSymbol = start.back();
    char playerSymbol = player.back();
    
    getBoard(cin);
    char cur = startSymbol;
    
    while (true) {
        char winner;
        if (isEndState(winner)) {
            if (winner == 'X') cout << "WINNER: X\n";
            else if (winner == 'O') cout << "WINNER: O\n";
            else cout << "DRAW\n";
            return 0;
        }
        
        if (cur == playerSymbol) {
            // Human move
            string moveLine;
            getline(cin, moveLine);
            int r = moveLine[0] - '0';
            int c = moveLine[2] - '0';
            applyMove(r, c, playerSymbol);
            printBoard(cout);
        } else {
            // AI move
            int move = eval(cur);
            if (move >= 0) {
                board[move] = cur;
            }
            printBoard(cout);
        }
        cur = (cur == 'X' ? 'O' : 'X');
    }
}
```

**Purpose**: Play full game, alternating between human and AI.

### 7. Board I/O (Lines 113-153)
```cpp
inline bool getBoard(istream& in) {
    string lines[7];
    for (int i = 0; i < 7; i++) {
        getline(in, lines[i]);
    }
    
    for (int r = 0; r < 3; r++) {
        string& line = lines[1 + r * 2];
        board[idx(r, 0)] = line[2];
        board[idx(r, 1)] = line[6];
        board[idx(r, 2)] = line[10];
    }
    return true;
}

inline void printBoard(ostream& out) {
    out << "+---+---+---+\n";
    for (int r = 0; r < 3; r++) {
        out << "| " << board[idx(r, 0)] << " | " 
            << board[idx(r, 1)] << " | " 
            << board[idx(r, 2)] << " |\n";
        out << "+---+---+---+\n";
    }
}
```

**Format**:
```
+---+---+---+
| X | _ | O |
+---+---+---+
| _ | X | _ |
+---+---+---+
| O | _ | X |
+---+---+---+
```

---

## Implementation Questions & Answers

### Q1: Where is the minimax algorithm?
**A**: Lines 40-88. Recursively evaluates all possible moves, returning best score and move.

### Q2: How does alpha-beta pruning work?
**A**: Lines 80-82:
```cpp
if (beta <= alpha) {
    break;  // Prune remaining branches
}
```

**Alpha**: Best score maximizer can guarantee.
**Beta**: Best score minimizer can guarantee.
**Prune when**: β ≤ α (opponent won't allow this branch).

### Q3: Where is the evaluation function?
**A**: Terminal states (lines 44-53):
- Win for root player: +1000 - depth
- Loss: -1000 + depth
- Draw: 0

### Q4: Why subtract/add depth?
**A**: **Prefer faster wins, slower losses**.

**Example**: Win in 2 moves scores 998, win in 5 moves scores 995. AI chooses 2-move win.

### Q5: How is the best move chosen?
**A**: Lines 66-78. For each position, try move, get score, keep best.

**Maximizing** (AI turn): Choose highest score.
**Minimizing** (opponent turn): Choose lowest score.

### Q6: What is the branching factor?
**A**: 
- First move: 9 options
- Average: ~5 options
- Decreases as game progresses

**Total game tree**: ~9! = 362,880 states (symmetry reduces this).

### Q7: Why INT_MIN/4 and INT_MAX/4 for alpha/beta?
**A**: Line 108. Prevent integer overflow when adding/subtracting during recursion.

**Alternative**: Could use special sentinel values.

### Q8: What happens on empty board?
**A**: Lines 99-104. Returns position 4 (center) immediately without search.

**Why center**: Optimal first move (most winning lines pass through it).

---

## Theoretical Questions & Answers

### Q1: What is the Minimax algorithm?
**A**: Decision rule for two-player zero-sum games.

**Idea**: 
- **Maximizer** (you): Maximize score
- **Minimizer** (opponent): Minimize your score
- Assume opponent plays optimally

**Formula**: $minimax(s) = \begin{cases} utility(s) & \text{if terminal} \\ \max_{a} minimax(result(s,a)) & \text{if maximizer} \\ \min_{a} minimax(result(s,a)) & \text{if minimizer} \end{cases}$

### Q2: What is alpha-beta pruning?
**A**: Optimization that eliminates branches that can't affect final decision.

**Alpha** (α): Best score maximizer can guarantee so far.
**Beta** (β): Best score minimizer can guarantee so far.

**Pruning rule**: If β ≤ α during minimizer's turn, or α ≥ β during maximizer's turn, prune remaining siblings.

**Benefit**: Reduces nodes from O(b^d) to O(b^(d/2)) in best case.

### Q3: Time and space complexity?
**A**: 
**Without pruning**: O(b^d) where b=branching factor (~5), d=depth (~9).
- Tic-tac-toe: ~5^9 ≈ 2 million nodes

**With pruning**: O(b^(d/2)) ≈ 5^4.5 ≈ 1,500 nodes (best case).

**Space**: O(d) for recursion stack.

**Tic-tac-toe**: Fast enough to search entire tree.

### Q4: Is minimax optimal?
**A**: **Yes**, assuming:
1. Opponent plays optimally
2. Complete game tree explored
3. Accurate evaluation function

**Tic-tac-toe**: All three conditions met → perfect play.

### Q5: What is a zero-sum game?
**A**: Game where one player's gain = other player's loss.

**Tic-tac-toe**: 
- X wins (+1) ⟹ O loses (-1)
- Draw (0) for both

**Not zero-sum**: Cooperative games where both can win/lose.

### Q6: What is a game tree?
**A**: Tree where:
- **Nodes**: Game states
- **Edges**: Moves
- **Leaves**: Terminal states (win/loss/draw)
- **Depth**: Moves from start

**Tic-tac-toe tree**: Root = empty board, leaves = game over.

### Q7: How does move ordering improve pruning?
**A**: **Better moves first → more pruning**.

**Example**: If evaluate best move first, all worse siblings can be pruned.

**This implementation**: No move ordering (could be improved).

**Advanced**: Evaluate center/corners first, or use iterative deepening.

### Q8: Compare minimax to other game algorithms.

| Algorithm | Optimal | Speed | Memory | Use Case |
|-----------|---------|-------|--------|----------|
| Minimax | Yes | O(b^d) | O(d) | Small trees |
| Alpha-beta | Yes | O(b^(d/2)) | O(d) | Medium trees |
| Monte Carlo Tree Search | Approximate | Anytime | O(nodes) | Large trees (Go) |
| Heuristic search | Approximate | Fast | O(d) | Chess, deep trees |

### Q9: Why is tic-tac-toe a solved game?
**A**: 
- **Finite states**: 3^9 = 19,683 positions (much less with symmetry)
- **Short games**: At most 9 moves
- **Complete search**: Minimax can explore entire tree in milliseconds

**Result**: Perfect play by both leads to draw.

### Q10: What is the evaluation function?
**A**: Function that estimates value of non-terminal state.

**Tic-tac-toe**: Not needed (searches to end).

**Chess**: Would use: material count, piece positions, king safety, etc.

**This implementation** (lines 44-53): Only evaluates terminal states.

### Q11: What are adversarial search problems?
**A**: Search problems with opponent.

**Components**:
- **Initial state**: Starting board
- **Players**: MAX and MIN alternate
- **Actions**: Legal moves
- **Transition**: Result of move
- **Terminal test**: Game over?
- **Utility**: Payoff for terminal state

**Tic-tac-toe**: Classic example.

### Q12: How would you extend this to larger games?
**A**: 
**Larger boards** (e.g., 4×4 tic-tac-toe):
1. **Depth limit**: Stop search at fixed depth
2. **Evaluation function**: Estimate non-terminal states
3. **Better pruning**: Move ordering, transposition tables
4. **Iterative deepening**: Search deeper over time

**Much larger** (e.g., chess):
1. All of above
2. Opening book: Pre-computed good starts
3. Endgame databases: Pre-solved end positions
4. Parallel search: Multiple threads

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Board representation | 5 |
| Index helper | 7-9 |
| Win detection | 11-20 |
| Terminal state check | 23-37 |
| Minimax with alpha-beta | 40-88 |
| Evaluation function | 91-110 |
| Board input | 113-128 |
| Board output | 130-139 |
| Apply move | 141-152 |
| Judge mode | 154-172 |
| Game mode | 174-215 |

---

## Performance Characteristics

**Tic-tac-toe specific**:
- Positions: 5,478 (after symmetry reduction)
- Search time: Milliseconds
- Perfect play: Always draws

**Alpha-beta benefit**: ~99% node reduction compared to naive minimax.

**Guaranteed outcome**: Draw with optimal play from both sides.
