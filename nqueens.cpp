#include <iostream>
#include <time.h>
#include <vector>
#include <random>
#include <chrono>
using namespace std;
static std::mt19937_64 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

int N;
vector<int> queenRowPerColumn;    // queen row per column
vector<int> numberOfQueensPerRow; // queens per row
vector<int> diag1;                // r - c + (N-1)
vector<int> diag2;                // r + c
vector<int> conflicted;           // conflicted queens
vector<int> posInConflicted;      // position in conflicted or -1

const int K = 10;               // steps per queen
const int MAX_RESTARTS = 20;    // maximum number of restarts
const int REFRESH = 100;        // refresh on every 100 steps to clear stale data
const int CORRECTION = 3;       // correction for self-conflicts
const int SAMPLE = 500;         // number of samples for large N
const int NEIGHBOR_RADIUS = 50; // radius for local search
const int BIG_N = 5000;         // threshold for large N


inline int d1(int r, int c) {
    return r - c + (N - 1);
}

inline int d2(int r, int c) {
    return r + c;
}

inline int conflictsAt(int r, int c) {
    return numberOfQueensPerRow[r] + diag1[d1(r, c)] + diag2[d2(r, c)];
}

inline int conflictsOfQueen(int c) {
    int r = queenRowPerColumn[c];
    return (numberOfQueensPerRow[r] - 1) + (diag1[d1(r, c)] - 1) + (diag2[d2(r, c)] - 1);
}

inline void tryRowForInitialization(int r, int c, int& optimalConflict, int& optimalRow) {
    int conflicts = conflictsAt(r, c);
    if (conflicts < optimalConflict) {
        optimalConflict = conflicts;
        optimalRow = r;
    }
    else if (conflicts == optimalConflict && (rng() & 1)) {
        optimalRow = r;
    }
}

inline void evaluateRowForMove(int r, int chosenColumn, int& optimalConflict, int& optimalRow) {
    int conflicts = conflictsAt(r, chosenColumn);
    if (r == queenRowPerColumn[chosenColumn]) {
        conflicts -= CORRECTION;
    }
    if (conflicts < optimalConflict) {
        optimalConflict = conflicts;
        optimalRow = r;
    }
    else if (conflicts == optimalConflict && (rng() & 1)) {
        optimalRow = r;
    }
}

inline void addConflicted(int c) {
    if (posInConflicted[c] == -1) {
        posInConflicted[c] = (int)conflicted.size();
        conflicted.push_back(c);
    }
}

inline void removeConflicted(int c) {
    int pos = posInConflicted[c];
    if (pos != -1) {
        int last = conflicted.back();
        conflicted[pos] = last;
        posInConflicted[last] = pos;
        conflicted.pop_back();
        posInConflicted[c] = -1;
    }
}

void initialize() {
    fill(queenRowPerColumn.begin(), queenRowPerColumn.end(), 0);
    fill(numberOfQueensPerRow.begin(), numberOfQueensPerRow.end(), 0);
    fill(diag1.begin(), diag1.end(), 0);
    fill(diag2.begin(), diag2.end(), 0);
    fill(posInConflicted.begin(), posInConflicted.end(), -1);
    conflicted.clear();

    // Optimization: randomized order initialization
    vector<int> order(N);
    for (int i = 0; i < N; i++) {
        order[i] = i;
    }
    shuffle(order.begin(), order.end(), rng);

    for (int i = 0; i < N; i++) {
        int c = order[i];
        int optimalRow = 0;
        int optimalConflict = INT_MAX;

        if (N <= BIG_N) {
            // full scan
            for (int r = 0; r < N; r++) {
                int conflicts = conflictsAt(r, c);
                if (conflicts < optimalConflict) {
                    optimalConflict = conflicts;
                    optimalRow = r;
                }
                else if (conflicts == optimalConflict && (rng() & 1)) {
                    optimalRow = r;
                }
            }
        }
        else {
            // Optimization: sampling
            tryRowForInitialization(c % N, c, optimalConflict, optimalRow);
            const int SAMPLE = 2000;
            for (int i = 0; i < SAMPLE; i++) {
                tryRowForInitialization((int)(rng() % N), c, optimalConflict, optimalRow);
            }
        }

        queenRowPerColumn[c] = optimalRow;
        numberOfQueensPerRow[optimalRow]++;
        diag1[d1(optimalRow, c)]++;
        diag2[d2(optimalRow, c)]++;
    }

    for (int c = 0; c < N; c++) {
        if (conflictsOfQueen(c) > 0) {
            addConflicted(c);
        }
    }
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N;

    bool print = false;
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--print-board") {
            print = true;
        }
    }

    bool timed = false;
    char* buff = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buff, &len, "FMI_timed") == 0 && buff != nullptr) {
        timed = (string(buff) == "1");
        free(buff);
    }

    // edge cases
    if (N == 1) {
        if (timed) {
            cout << "# TIMES_MS: alg=0\n";
            return 0;
        }
        cout << "[0]\n";
        if (print) cout << "*\n";
        return 0;
    }

    if (N == 2 || N == 3) {
        if (timed) {
            cout << "# TIMES_MS: alg=0\n";
        }
        else {
            cout << -1 << '\n';
        }
        return 0;
    }

    auto t0 = chrono::high_resolution_clock::now();

    queenRowPerColumn.assign(N, 0);
    numberOfQueensPerRow.assign(N, 0);
    diag1.assign(2 * N - 1, 0);
    diag2.assign(2 * N - 1, 0);
    conflicted.clear();
    posInConflicted.assign(N, -1);

    bool solved = false;

    for (int restart = 0; restart < MAX_RESTARTS && !solved; restart++) {
        initialize();
        int stepsLimit = K * N;

        for (int step = 0; step < stepsLimit && !conflicted.empty(); step++) {
            int maxConflicts = -1;
            vector<int> candidates;
            candidates.reserve(16);

            for (int c : conflicted) {
                int conflict = conflictsOfQueen(c);
                if (conflict > maxConflicts) {
                    maxConflicts = conflict;
                    candidates.clear();
                    candidates.push_back(c);
                }
                else if (conflict == maxConflicts) {
                    candidates.push_back(c);
                }
            }

            if (maxConflicts <= 0) {
                break;
            }

            // Optimization: random among max conflicted
            int chosenColumn = candidates[(size_t)(rng() % candidates.size())];

            // find best row for chosenColumn
            int optimalRow = queenRowPerColumn[chosenColumn];
            int optimalConflict = INT_MAX;

            if (N <= BIG_N) {
                for (int r = 0; r < N; r++) {
                    int conflicts = conflictsAt(r, chosenColumn);
                    if (r == queenRowPerColumn[chosenColumn]) {
                        conflicts -= CORRECTION;
                    }
                    if (conflicts < optimalConflict) {
                        optimalConflict = conflicts;
                        optimalRow = r;
                    }
                    else if (conflicts == optimalConflict && (rng() & 1)) {
                        optimalRow = r;
                    }
                }
            }
            else {
                // Optimization: sampling + local search via neighborhood
                evaluateRowForMove(queenRowPerColumn[chosenColumn], chosenColumn, optimalConflict, optimalRow);
                evaluateRowForMove(chosenColumn % N, chosenColumn, optimalConflict, optimalRow);

                for (int i = 0; i < SAMPLE; i++) {
                    evaluateRowForMove((int)(rng() % N), chosenColumn, optimalConflict, optimalRow);
                }

                // Optimization: local search in neighborhood with radius 50
                for (int i = -NEIGHBOR_RADIUS; i <= NEIGHBOR_RADIUS; i++) {
                    evaluateRowForMove((queenRowPerColumn[chosenColumn] + i + N) % N, chosenColumn, optimalConflict, optimalRow);
                }
            }

            int oldRow = queenRowPerColumn[chosenColumn];
            if (optimalRow != oldRow) {
                numberOfQueensPerRow[oldRow]--;
                diag1[d1(oldRow, chosenColumn)]--;
                diag2[d2(oldRow, chosenColumn)]--;
                queenRowPerColumn[chosenColumn] = optimalRow;
                numberOfQueensPerRow[optimalRow]++;
                diag1[d1(optimalRow, chosenColumn)]++;
                diag2[d2(optimalRow, chosenColumn)]++;
            }

            if (conflictsOfQueen(chosenColumn) == 0) {
                removeConflicted(chosenColumn);
            }
            else {
                addConflicted(chosenColumn);
            }

            // full refresh
            if (step % REFRESH == 0) {
                for (int c = 0; c < N; c++) {
                    if (conflictsOfQueen(c) > 0) {
                        addConflicted(c);
                    }
                    else {
                        removeConflicted(c);
                    }
                }
            }
            else {
                // smaller refresh, since it could reduce stale data, so 3 random queens
                for (int i = 0; i < 3; i++) {
                    int c = (int)(rng() % N);
                    if (conflictsOfQueen(c) > 0) {
                        addConflicted(c);
                    }
                    else {
                        removeConflicted(c);
                    }
                }
            }
        }

        bool ok = true;
        for (int c = 0; c < N; c++)
            if (conflictsOfQueen(c) != 0) {
                ok = false;
                break;
            }

        if (ok) {
            solved = true;
            break;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    long long ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    if (timed) {
        cout << "# TIMES_MS: alg=" << ms << '\n';
        return 0;
    }

    if (!solved) {
        cout << -1 << '\n';
        return 0;
    }

    cout << '[';
    for (int c = 0; c < N; c++) {
        if (c) {
            cout << ", ";
        }
        cout << queenRowPerColumn[c];
    }
    cout << ']' << '\n';

    if (print) {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                cout << (queenRowPerColumn[c] == r ? '*' : '_');
                if (c + 1 < N) {
                    cout << ' ';
                }
            }
            cout << '\n';
        }
    }

    return 0;
}
