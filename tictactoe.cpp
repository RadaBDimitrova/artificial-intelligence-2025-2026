#include <iostream>
#include <string>
#include <climits>
using namespace std;
char board[9] = {'_','_','_','_','_','_','_','_','_'};

inline int idx(int r, int c) {
    return (r * 3 + c);
}

inline bool isWinner(char p) {
    int wins[8][3] = { {0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6} };
    for (int i = 0; i < 8; i++) {
        if (board[wins[i][0]] == p && board[wins[i][1]] == p && board[wins[i][2]] == p) {
            return true;
        }
    }
    return false;
}

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
            return false;
        }
    }

    winner = '_';
    return true;
}

pair<int, int> minimax(char rootMove, char currentMove, int depth, int alpha, int beta) {
    char winner;

    if (isEndState(winner)) {
        if (winner == rootMove) {
            return { 1000 - depth, -1 };
        }
        else if (winner == '_') {
            return { 0, -1 };
        }
        else {
            return { -1000 + depth, -1 };
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
            }
            else {
                if (sc < bestScore) {
                    bestScore = sc;
                    bestMove = i;
                }
                beta = min(beta, bestScore);
            }

            if (beta <= alpha) {
                break;
            }
        }
    }
    return { bestScore, bestMove };
}

inline int eval(char playerChar) {
    char winner;
    if (isEndState(winner)) {
        return -1;
    }

    bool isEmpty = true;
    for (int i = 0; i < 9; i++) {
        if (board[i] != '_') {
            isEmpty = false;
            break;
        }
    }
    if (isEmpty) {
        return 4;
    }

    auto res = minimax(playerChar, playerChar, 0, INT_MIN / 4, INT_MAX / 4);
    return res.second;
}

inline bool getBoard(istream& in) {
    string lines[7];
    for (int i = 0; i < 7; i++) {
        getline(in, lines[i]);
        if (!lines[i].empty() && lines[i].back() == '\r') {
            lines[i].pop_back();
        }
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
        out << "| ";
        out << board[idx(r, 0)] << " | ";
        out << board[idx(r, 1)] << " | ";
        out << board[idx(r, 2)] << " |" << "\n";
        out << "+---+---+---+\n";
    }
    out.flush();
}

inline bool applyMove(int row, int col, char player) {
    if (row < 1 || row > 3 || col < 1 || col > 3) {
        return false;
    }
    int id = idx(row - 1, col - 1);
    if (board[id] != '_') {
        return false;
    }
    board[id] = player;
    return true;
}

int main() {
    string mode;
    getline(cin, mode);

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
            return 0;
        } 
        else {
            int rr = (move / 3) + 1;
            int cc = (move % 3) + 1;
            cout << rr << " " << cc << "\n";
            return 0;
        }
    } 
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
                if (winner == 'X') {
                    cout << "WINNER: X\n";
                }
                else if (winner == 'O') {
                    cout << "WINNER: O\n";
                }
                else {
                    cout << "DRAW\n";
                }
                return 0;
            }

            if (cur == playerSymbol) {
                string moveLine;
                getline(cin, moveLine);
                int r = moveLine[0] - '0';
                int c = moveLine[2] - '0';
                applyMove(r, c, playerSymbol);
                printBoard(cout);
            }
            else {
                int move = eval(cur);
                if (move >= 0) {
                    board[move] = cur;
                }
                printBoard(cout);
            }
            cur = (cur == 'X' ? 'O' : 'X');
        }
    }
    return 0;
}
