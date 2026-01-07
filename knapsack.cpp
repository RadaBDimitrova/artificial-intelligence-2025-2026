#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
using namespace std;

static mt19937_64 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());

struct Item {
    int w;
    int v;
};

int N;
long long M;
vector<Item> items;

const int POP = 200;
const double MUTATION_RATE = 0.02;
const int ELITE_COUNT = 4;
const double MAX_TIME = 1.5;
const int RESULTS_NEEDED = 10;

struct Individual {
    vector<uint8_t> genes;
    long long weight;
    long long value;
};

vector<Individual> pop(POP); // population
vector<long long> results;  // fitness results

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

// Tournament Selection
inline int tournamentSelection() {
    int a = rng() % POP, b = rng() % POP;
    return (pop[a].value > pop[b].value) ? a : b;
}

// Fitness Proportionate Selection  and "Natural" Selection
inline void rouletteSelection(Individual& ind) {
    if (ind.weight <= M) {
        return;
    }
    // removing smallest v/w ratio first when overweight
    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        double ra = (double)items[a].v / items[a].w;
        double rb = (double)items[b].v / items[b].w;
        return ra < rb;
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

// one-point crossover
// note: would uniform + one/two-point crossover fit this also
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

// mutation with 2% rate
inline void mutate(Individual& ind) {
    for (int i = 0; i < N; i++) {
        if (((double)rng() / rng.max()) < MUTATION_RATE) {
            ind.genes[i] = 1 - ind.genes[i];
        } 
    }
}

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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> M >> N;
    items.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> items[i].w >> items[i].v;
    }

    initialize();
    auto t0 = chrono::high_resolution_clock::now();

    while (true) {
        auto t1 = chrono::high_resolution_clock::now();
        if (chrono::duration<double>(t1 - t0).count() > MAX_TIME) {
            break;
        }

        sort(pop.begin(), pop.end(), [](const Individual& a, const Individual& b) {
             return a.value > b.value;
        });
        results.push_back(pop[0].value);

        vector<Individual> newPop;
        for (int i = 0; i < ELITE_COUNT; i++) {
            newPop.push_back(pop[i]); // memoisation and elitism
        }

        for (int i = ELITE_COUNT; i < POP; i++) {
            int a = tournamentSelection();
            int b = tournamentSelection();
            Individual child = crossover(pop[a], pop[b]);
            mutate(child);
            evaluate(child);
            rouletteSelection(child);
            newPop.push_back(child);
        }

        pop.swap(newPop);
    }

    vector<long long> finalResults;
    int size = results.size();
    if (size >= RESULTS_NEEDED) {
        for (int i = 0; i < RESULTS_NEEDED; i++) {
            int idx = (int)round((double)i * (size - 1) / (RESULTS_NEEDED - 1));
            finalResults.push_back(results[idx]);
        }
    }
    else {
        finalResults = results;
        while ((int)finalResults.size() < RESULTS_NEEDED) {
            finalResults.push_back(results.back());
        }
    }

    for (long long v : finalResults) {
        cout << v << "\n";
    }
    cout << "\n" << finalResults.back() << "\n";
    return 0;
}
