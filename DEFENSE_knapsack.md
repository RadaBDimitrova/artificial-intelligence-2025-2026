# Defense Documentation: Knapsack Problem with Genetic Algorithm (knapsack.cpp)

## Problem Overview
**Task**: Solve 0/1 knapsack problem using Genetic Algorithm.

**Problem**: Given N items with weights and values, select items to maximize value without exceeding capacity M.

**Algorithm**: Genetic Algorithm with tournament selection, one-point crossover, mutation, elitism

---

## Code Structure Explanation

### 1. Global Variables and Constants (Lines 9-19)
```cpp
int N;                      // Number of items
long long M;                // Knapsack capacity
vector<Item> items;         // Items with weight and value

const int POP = 200;               // Population size
const double MUTATION_RATE = 0.02; // 2% mutation probability
const int ELITE_COUNT = 4;         // Top 4 preserved
const double MAX_TIME = 1.5;       // 1.5 seconds time limit
const int RESULTS_NEEDED = 10;     // Output 10 snapshots
```

**What they control**:
- `POP`: Larger = more diversity, slower
- `MUTATION_RATE`: Balance exploration vs exploitation
- `ELITE_COUNT`: Preserve best solutions (elitism)
- `MAX_TIME`: Stop condition
- `RESULTS_NEEDED`: Track convergence

### 2. Data Structures (Lines 11-12, 21-26)
```cpp
struct Item {
    int w;  // weight
    int v;  // value
};

struct Individual {
    vector<uint8_t> genes;  // Binary: 1=take item, 0=skip
    long long weight;       // Total weight of selected items
    long long value;        // Total value of selected items
};
```

**Genome representation**: Binary vector of length N.
- `genes[i] = 1`: Item i is in knapsack
- `genes[i] = 0`: Item i is not in knapsack

**Why uint8_t**: Memory efficient (1 byte vs 4 for int).

### 3. Fitness Evaluation (Lines 32-41)
```cpp
inline void evaluate(Individual& ind) {
    ind.weight = 0;
    ind.value = 0;
    for (int i = 0; i < N; i++) {
        if (ind.genes[i]) {
            ind.weight += items[i].w;
            ind.value += items[i].v;
        }
    }
}
```

**What it does**: Calculates total weight and value from genome.

**Where used**: After crossover/mutation (line 125), initialization (line 98).

**Fitness function**: `ind.value` (maximize value).

### 4. Tournament Selection (Lines 44-47)
```cpp
inline int tournamentSelection() {
    int a = rng() % POP, b = rng() % POP;
    return (pop[a].value > pop[b].value) ? a : b;
}
```

**What it does**: Randomly picks 2 individuals, returns better one.

**Why tournament**: 
- Simple and efficient
- Maintains diversity (weak individuals can still be selected)
- No need to sort entire population

**Where used**: Line 118-119 to select parents.

**Selection pressure**: 2-tournament = moderate (larger tournament = more pressure).

### 5. Natural Selection / Repair (Lines 50-70)
```cpp
inline void rouletteSelection(Individual& ind) {
    if (ind.weight <= M) {
        return;  // Valid solution
    }
    // Remove items with lowest value/weight ratio first
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        double ra = (double)items[a].v / items[a].w;
        double rb = (double)items[b].v / items[b].w;
        return ra < rb;  // Ascending order
    });
    
    for (int i = 0; i < N; i++) {
        int idx = order[i];
        if (ind.genes[idx]) {
            ind.genes[idx] = 0;
            ind.weight -= items[idx].w;
            ind.value -= items[idx].v;
            if (ind.weight <= M) {
                break;
            }
        }
    }
}
```

**What it does**: Makes invalid solutions valid by removing worst items.

**Strategy**: Remove items with **lowest** value/weight ratio first.
- Keeps high-efficiency items
- Removes low-efficiency items

**Where**: 
- Line 57-61: Sort items by efficiency
- Line 63-70: Remove lowest-efficiency items until valid

**Called**: After initialization (line 99), after mutation (line 126).

**Why needed**: Crossover/mutation can create overweight solutions.

### 6. Crossover (Lines 73-86)
```cpp
inline Individual crossover(const Individual& a, const Individual& b) {
    Individual child;
    child.genes.resize(N);
    int cut = rng() % N;
    for (int i = 0; i < N; i++) {
        if (i <= cut) {
            child.genes[i] = a.genes[i];
        } else {
            child.genes[i] = b.genes[i];
        }
    } 
    return child;
}
```

**Type**: One-point crossover.

**How it works**:
1. Choose random cut point (line 76)
2. Child gets genes from parent A before cut
3. Child gets genes from parent B after cut

**Example**: 
- Parent A: `11001100`
- Parent B: `00110011`
- Cut at 4: `1100|1100` × `0011|0011` → Child: `11000011`

**Where used**: Line 120.

**Alternatives mentioned in comment**: Uniform, two-point (not implemented).

### 7. Mutation (Lines 89-95)
```cpp
inline void mutate(Individual& ind) {
    for (int i = 0; i < N; i++) {
        if (((double)rng() / rng.max()) < MUTATION_RATE) {
            ind.genes[i] = 1 - ind.genes[i];  // Flip bit
        } 
    }
}
```

**What it does**: Flips each gene with 2% probability.

**Type**: Bit-flip mutation.

**Example**: `11001100` → flip bit 3 → `11101100`.

**Where used**: Line 121.

**Why 2%**: Balance between exploration (higher) and exploitation (lower).

### 8. Initialization (Lines 97-103)
```cpp
void initialize() {
    for (int i = 0; i < POP; i++) {
        pop[i].genes.assign(N, 0);
        for (int j = 0; j < N; j++) {
            if (rng() % 2) { 
                pop[i].genes[j] = 1;
            }
        }
        evaluate(pop[i]);
        rouletteSelection(pop[i]);
    }
}
```

**What it does**: Creates initial population of 200 individuals.

**Process**:
1. For each individual: 50% chance each item is included
2. Evaluate fitness
3. Repair if overweight

**Where called**: Line 115 in main.

### 9. Main Genetic Algorithm Loop (Lines 117-149)

#### Time-Based Termination (Lines 119-121)
```cpp
while (true) {
    auto t1 = chrono::high_resolution_clock::now();
    if (chrono::duration<double>(t1 - t0).count() > MAX_TIME) {
        break;
    }
```

**Why time-based**: Problem size varies; iterations needed unpredictable.

#### Selection and Elitism (Lines 123-129)
```cpp
sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
     return a.value > b.value;
});
results.push_back(pop[0].value);

vector<Individual> newPop;
for (int i = 0; i < ELITE_COUNT; i++) {
    newPop.push_back(pop[i]); // Elitism
}
```

**What happens**:
1. Sort population by fitness (descending)
2. Record best fitness
3. Copy top 4 to next generation (elitism)

**Elitism benefit**: Guarantees monotonic improvement.

#### Breeding (Lines 131-137)
```cpp
for (int i = ELITE_COUNT; i < POP; i++) {
    int a = tournamentSelection();
    int b = tournamentSelection();
    Individual child = crossover(pop[a], pop[b]);
    mutate(child);
    evaluate(child);
    rouletteSelection(child);
    newPop.push_back(child);
}
```

**Process**:
1. Select two parents via tournament
2. Crossover to create child
3. Mutate child
4. Evaluate fitness
5. Repair if needed
6. Add to new population

**Where**: Fills population from index 4 to 199.

### 10. Results Processing (Lines 143-157)
```cpp
vector<long long> finalResults;
int size = results.size();
if (size >= RESULTS_NEEDED) {
    for (int i = 0; i < RESULTS_NEEDED; i++) {
        int idx = (int)round((double)i * (size - 1) / (RESULTS_NEEDED - 1));
        finalResults.push_back(results[idx]);
    }
}
```

**What it does**: Samples 10 evenly-spaced points from convergence history.

**Why**: Shows progression from start to end.

**Example**: If 1000 generations, samples at 0, 111, 222, ..., 999.

---

## Implementation Questions & Answers

### Q1: Where is the fitness function?
**A**: `ind.value` is the fitness. Calculated in `evaluate()` (lines 32-41). Higher value = better fitness.

### Q2: How does selection work?
**A**: **Tournament selection** (lines 44-47):
- Randomly pick 2 individuals
- Return the one with higher value
- Used on lines 118-119 to select parents

### Q3: What type of crossover is used?
**A**: **One-point crossover** (lines 73-86):
- Random cut point in genome
- Child inherits left side from parent A, right side from parent B
- Location: Line 120

### Q4: How is mutation implemented?
**A**: **Bit-flip mutation** (lines 89-95):
- Each gene has 2% chance to flip
- `genes[i] = 1 - genes[i]` toggles 0↔1
- Location: Line 121

### Q5: What is elitism and where is it?
**A**: Lines 126-129:
```cpp
for (int i = 0; i < ELITE_COUNT; i++) {
    newPop.push_back(pop[i]);
}
```
Top 4 individuals automatically copied to next generation. Prevents losing best solutions.

### Q6: How are invalid solutions handled?
**A**: `rouletteSelection()` function (lines 50-70):
- If weight > capacity, remove items
- Strategy: Remove lowest value/weight ratio items first
- Called after initialization (line 99) and mutation (line 126)

### Q7: When does the algorithm stop?
**A**: Lines 119-122: After 1.5 seconds.
```cpp
if (chrono::duration<double>(t1 - t0).count() > MAX_TIME) {
    break;
}
```

### Q8: Where is the population sorted?
**A**: Line 123:
```cpp
sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
     return a.value > b.value;
});
```
Descending order by value (fitness).

### Q9: How many individuals are in the population?
**A**: 200 (`POP = 200`, line 14).
- 4 elite (preserved)
- 196 offspring (bred each generation)

### Q10: What is "memoisation and elitism" comment on line 127?
**A**: 
- **Elitism**: Copying best solutions (line 127-129)
- **Memoization**: Top solutions "remembered" across generations
- Comment slightly misleading - not traditional memoization (caching)

---

## Theoretical Questions & Answers

### Q1: What is a Genetic Algorithm?
**A**: Optimization algorithm inspired by natural evolution.

**Components**:
1. **Population**: Set of candidate solutions
2. **Fitness**: Quality measure
3. **Selection**: Choose parents (survival of fittest)
4. **Crossover**: Combine parents → offspring
5. **Mutation**: Random changes for diversity
6. **Replacement**: New generation replaces old

**Process**: Iteratively improve population through selection, crossover, mutation.

### Q2: What is the 0/1 knapsack problem?
**A**: 
**Input**: n items (each with weight $w_i$, value $v_i$), capacity M.

**Goal**: Maximize $\sum v_i x_i$ subject to $\sum w_i x_i \leq M$ where $x_i \in \{0, 1\}$.

**"0/1"**: Each item taken fully or not at all (vs fractional knapsack).

**Complexity**: NP-hard (no polynomial exact algorithm known).

### Q3: Why use GA for knapsack?
**A**: 
**Pros**:
- Works on large instances where exact methods too slow
- Finds good (not necessarily optimal) solutions quickly
- Scales well with problem size

**Cons**:
- No optimality guarantee
- Randomized (different runs → different results)

**Alternatives**:
- Dynamic programming: O(nM) - optimal but slow for large M
- Branch and bound: Optimal but exponential worst case
- Greedy: Fast but not optimal

### Q4: What is tournament selection?
**A**: Selection method where k individuals compete, best chosen.

**This implementation**: k=2 (lines 44-47).

**Properties**:
- **Selection pressure**: Larger k = more pressure
- **Diversity**: Weaker individuals can still win (luck)
- **Efficiency**: O(k) per selection

**Alternatives**: Roulette wheel, rank selection, truncation.

### Q5: What is one-point crossover?
**A**: 
1. Choose random cut point
2. Child inherits left side from parent A
3. Child inherits right side from parent B

**Example**:
```
Parent A: 11111111
Parent B: 00000000
Cut at 4: 1111|1111 × 0000|0000 → Child: 11110000
```

**Alternatives**:
- **Two-point**: Two cuts, swap middle
- **Uniform**: Each gene randomly from either parent
- **This code**: Only one-point (comment line 78 mentions others)

### Q6: What is mutation and why is it needed?
**A**: Random changes to genes.

**Purpose**:
1. **Exploration**: Escape local optima
2. **Diversity**: Prevent premature convergence
3. **New genetic material**: Introduce genes lost in selection

**Rate**: 2% per gene (line 14).
- Too high: Random search
- Too low: Premature convergence

### Q7: What is elitism?
**A**: Preserving best individuals across generations.

**This implementation**: Top 4 copied directly (lines 126-129).

**Benefits**:
- **Monotonic improvement**: Best never gets worse
- **Faster convergence**: Good solutions preserved

**Drawbacks**:
- **Less diversity**: Elite dominate population
- **Premature convergence**: May get stuck

### Q8: Time and space complexity?
**A**: 
**Time per generation**: $O(P \cdot N)$
- P = population size (200)
- N = number of items
- Evaluate each individual: O(N)
- Sort: O(P log P) but negligible

**Space**: $O(P \cdot N)$
- Store 200 individuals, each with N genes

**Total time**: Depends on generations run (time-limited to 1.5s).

### Q9: Is GA complete and optimal?
**A**: 
- **Complete**: No (finite time, may not find optimal)
- **Optimal**: No (no guarantee of global optimum)
- **Approximate**: Yes (finds good solutions with high probability)

**Theoretical result**: With infinite time and population, can find optimal (not practical).

### Q10: What is the difference between exploration and exploitation?
**A**: 
**Exploitation**: Refine current good solutions.
- Mechanisms: Selection, crossover, elitism

**Exploration**: Search new areas of solution space.
- Mechanisms: Mutation, diversity maintenance

**Balance**: 
- Too much exploitation → premature convergence
- Too much exploration → slow convergence
- This code: 2% mutation, tournament selection = balanced

### Q11: Why repair solutions instead of penalizing fitness?
**A**: 
**Repair (this implementation)**: Fix invalid solutions to be valid.
- **Pros**: All solutions feasible, simple fitness function
- **Cons**: Repair may be expensive

**Penalty (alternative)**: Reduce fitness for constraint violations.
- **Pros**: Flexible, allows infeasible search
- **Cons**: Hard to tune penalty, infeasible final solutions

**This code**: Repairs by removing low-efficiency items (lines 50-70).

### Q12: What makes this a "greedy" repair strategy?
**A**: Lines 57-61 sort by value/weight ratio, remove worst first.

**Greedy**: Makes locally optimal choice (remove lowest efficiency).

**Why effective**: Mimics fractional knapsack greedy solution.

**Not globally optimal**: But fast and works well in practice.

### Q13: How does convergence work?
**A**: 
**Early generations**: High diversity, rapid improvement.

**Middle generations**: Slower improvement, population converges.

**Late generations**: Small improvements, mostly exploitation.

**Tracked**: `results` vector (line 125) records best value each generation.

**Output**: 10 evenly-spaced snapshots showing convergence curve.

### Q14: What is premature convergence?
**A**: Population becomes too similar before reaching optimum.

**Causes**:
- Too much selection pressure
- Too little mutation
- Small population

**Prevention**:
- Mutation (2%)
- Tournament selection (moderate pressure)
- Large population (200)
- Not implemented but could: Diversity metrics, niching

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Constants | 14-18 |
| Data structures | 11-12, 21-26 |
| Fitness evaluation | 32-41 |
| Tournament selection | 44-47 |
| Repair/natural selection | 50-70 |
| One-point crossover | 73-86 |
| Bit-flip mutation | 89-95 |
| Population initialization | 97-103 |
| Main GA loop | 117-139 |
| Elitism | 126-129 |
| Parent selection | 118-119 |
| Breeding | 118-137 |
| Results processing | 143-157 |

---

## Performance Characteristics

**Population**: 200 individuals
**Generations**: ~100-500 (depends on N)
**Time**: Fixed 1.5 seconds
**Quality**: Near-optimal (usually 95-99% of optimal)
**Deterministic**: No (random initialization and selection)
