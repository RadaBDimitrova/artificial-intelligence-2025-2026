# General AI & Machine Learning Theory - Defense Questions

## Overview
This document covers cross-cutting theoretical questions that apply to multiple tasks in the course. Expect at least one question from each category.

---

## 1. Supervised vs. Unsupervised Learning

### Q1: What is supervised learning?
**A**: Learning from labeled training data.

**Components**:
- **Input**: Features X
- **Output**: Labels y
- **Goal**: Learn mapping f: X → y
- **Training**: Use (X, y) pairs
- **Evaluation**: Predict labels for unseen data

**Examples in course**:
- **ID3**: (features, class) → decision tree
- **K-NN**: (features, class) → classify by neighbors
- **Naive Bayes**: (votes, party) → probability model

**Key characteristic**: Have "ground truth" labels.

### Q2: What is unsupervised learning?
**A**: Learning from unlabeled data - discovering patterns without explicit targets.

**Goal**: Find structure in data (clusters, dimensions, density).

**Examples in course**:
- **K-means**: Group similar points into clusters

**Types**:
- **Clustering**: Group similar items (K-means)
- **Dimensionality reduction**: PCA, t-SNE
- **Density estimation**: Find data distribution
- **Anomaly detection**: Find outliers

**Key characteristic**: No "correct answer" provided.

### Q3: Compare supervised and unsupervised learning.

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| **Labels** | Required | Not needed |
| **Goal** | Predict labels | Find patterns |
| **Evaluation** | Accuracy, F1, etc. | Silhouette, WCSS |
| **Examples** | Classification, regression | Clustering, PCA |
| **Difficulty** | Easier to evaluate | Subjective quality |
| **Data needs** | Labeled (expensive) | Unlabeled (cheap) |

### Q4: What is semi-supervised learning?
**A**: Mix of labeled and unlabeled data.

**Scenario**: Few labeled samples, many unlabeled.

**Approach**: Use unlabeled data to improve model trained on labeled data.

**Not in course**, but good to know.

### Q5: What is reinforcement learning?
**A**: Learning by interaction - agent learns through trial and error.

**Components**:
- **Agent**: Learner
- **Environment**: World
- **Actions**: Choices
- **Rewards**: Feedback
- **Goal**: Maximize cumulative reward

**Examples**: Game AI, robotics, AlphaGo.

**Not in course**, but related to search (game trees).

---

## 2. Overfitting vs. Underfitting

### Q6: What is overfitting?
**A**: Model learns training data too well, including noise; poor generalization.

**Signs**:
- High training accuracy
- Low test accuracy
- Large gap between train and test

**Causes**:
- Model too complex
- Too many features
- Too little data
- Training too long
- No regularization

**In course examples**:
- **ID3**: Deep tree memorizes training data
- **K-NN**: k=1 overfits (memorizes training examples)
- **Genetic algorithm**: Population converges too early

### Q7: What is underfitting?
**A**: Model too simple to capture data patterns.

**Signs**:
- Low training accuracy
- Low test accuracy
- Both poor, similar values

**Causes**:
- Model too simple
- Too few features
- Too much regularization
- Insufficient training

**In course examples**:
- **ID3**: Tree depth too limited
- **K-NN**: k too large (smooths out boundaries)

### Q8: How do you prevent overfitting?

**General techniques**:
1. **More data**: Dilutes noise
2. **Regularization**: Penalty for complexity (L1, L2)
3. **Cross-validation**: Detect overfitting early
4. **Early stopping**: Stop before over-learning
5. **Ensemble methods**: Average multiple models
6. **Feature selection**: Remove irrelevant features

**In course implementations**:
- **ID3**: 
  - Pre-pruning: max_depth, min_samples, min_gain
  - Post-pruning: Chi-squared test
  - Cross-validation: Detect overfitting
- **K-NN**: 
  - Cross-validation: Choose optimal k
  - K>1: Smooth decision boundaries
- **Naive Bayes**: 
  - Laplace smoothing: Prevents zero probabilities
- **K-means**: 
  - Multiple random restarts: Avoid local minima
  - Evaluation metrics: Detect poor clustering

### Q9: Visualize overfitting, good fit, and underfitting.

**Concept**:
```
Underfitting:  ___/‾‾‾\___     (too smooth, misses patterns)
Good fit:      __/‾\_/‾\__     (captures real patterns)
Overfitting:   _/‾\_/‾\_/‾\    (fits noise, too complex)
```

**Training vs. Model Complexity**:
- **Underfit**: Both train and test error high
- **Good**: Both train and test error low, similar
- **Overfit**: Train error low, test error high

### Q10: What is the bias-variance tradeoff?
**A**: Fundamental tradeoff in ML.

**Bias**: Error from incorrect assumptions.
- High bias → underfitting
- Example: Linear model for nonlinear data

**Variance**: Error from sensitivity to training data.
- High variance → overfitting
- Example: Memorizing training noise

**Formula**: $Error = Bias^2 + Variance + Noise$

**Goal**: Balance both.

**In course**:
- **Simple model** (high bias, low variance): Linear, large k in K-NN
- **Complex model** (low bias, high variance): Deep tree, k=1 in K-NN

---

## 3. Evaluation Metrics & Experimental Setup

### Q11: What is cross-validation?
**A**: Technique to assess model generalization using training data.

**K-Fold CV**:
1. Split data into k folds
2. For each fold i:
   - Train on k-1 folds
   - Validate on fold i
3. Average k accuracy scores

**Benefits**:
- More robust than single train/test split
- Uses all data for training and validation
- Reduces variance in evaluation

**In course**: All supervised learning tasks use 10-fold CV.

**Stratified CV**: Maintains class proportions in each fold (used in course).

### Q12: Why stratified sampling?
**A**: Preserves class distribution in splits.

**Example**: 70% class A, 30% class B.
- **Stratified**: Each fold has 70/30 split
- **Random**: Could vary widely (85/15, 55/45, etc.)

**Benefit**: More reliable evaluation, especially with:
- Imbalanced classes
- Small datasets

**In course**: Used in ID3, K-NN, Naive Bayes.

### Q13: What is train/test split?
**A**: Dividing data into separate training and test sets.

**Typical split**: 80/20 or 70/30.

**Purpose**:
- **Train**: Learn model parameters
- **Test**: Evaluate generalization (simulates unseen data)

**Critical**: Test set never seen during training.

**In course**: All supervised tasks use stratified 80/20 split.

### Q14: What is accuracy and when is it misleading?
**A**: 
**Accuracy**: $\frac{\text{correct predictions}}{\text{total predictions}}$

**Misleading when**: Imbalanced classes.

**Example**: 95% class A, 5% class B.
- Model predicting all A: 95% accuracy!
- But never predicts B correctly.

**Better metrics** for imbalanced:
- **Precision**: Of predicted positives, how many correct?
- **Recall**: Of actual positives, how many found?
- **F1 score**: Harmonic mean of precision and recall
- **Confusion matrix**: Full breakdown

**In course**: Accuracy used (classes fairly balanced).

### Q15: What is a confusion matrix?
**A**: Table showing prediction performance.

```
                Predicted
              Pos    Neg
Actual  Pos   TP     FN
        Neg   FP     TN
```

**Metrics from matrix**:
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: 2 × (Precision × Recall) / (Precision + Recall)

**Not explicitly in course**, but good to understand.

### Q16: What is a baseline?
**A**: Simple reference model to compare against.

**Common baselines**:
- **Random guessing**: 50% for binary, 33% for 3-class
- **Majority class**: Always predict most common class
- **Random chance**: Accuracy from random predictions

**Purpose**: Show model actually learns (beats baseline).

**Example**: Iris has 3 classes → random baseline = 33%. K-NN gets 95% → much better!

---

## 4. Search Algorithms (General)

### Q17: Compare uninformed vs. informed search.

| Aspect | Uninformed | Informed |
|--------|------------|----------|
| **Heuristic** | No | Yes |
| **Examples** | BFS, DFS, UCS | A*, IDA*, Greedy |
| **Optimality** | Depends (BFS yes, DFS no) | Depends on heuristic |
| **Speed** | Slower | Faster (with good h) |
| **In course** | DFS (frog-leap) | IDA* (n-puzzle) |

### Q18: What makes a heuristic admissible?
**A**: **Never overestimates** true cost to goal.

**Formula**: For all n, $h(n) \leq h^*(n)$ where $h^*(n)$ is true cost.

**Why important**: Guarantees A*/IDA* optimality.

**In course**: Manhattan distance for n-puzzle is admissible.

### Q19: What makes a heuristic consistent (monotonic)?
**A**: **Triangle inequality**: $h(n) \leq c(n, n') + h(n')$

**Meaning**: Heuristic decrease ≤ actual cost of move.

**Relationship**: Consistent → Admissible (but not vice versa).

**Benefit**: Nodes never reopened in A*.

**In course**: Manhattan distance is consistent.

### Q20: Compare completeness and optimality.

**Complete**: Guaranteed to find solution if one exists.

**Optimal**: Guaranteed to find best (lowest cost) solution.

| Algorithm | Complete | Optimal |
|-----------|----------|---------|
| DFS | No (infinite loops) | No |
| BFS | Yes | Yes (unweighted) |
| UCS | Yes | Yes |
| A* | Yes | Yes (admissible h) |
| IDA* | Yes | Yes (admissible h) |
| Greedy | No | No |
| Hill climbing | No | No |

---

## 5. Algorithm Comparisons

### Q21: Compare local search vs. systematic search.

| Aspect | Local Search | Systematic Search |
|--------|--------------|-------------------|
| **Memory** | O(1) or O(k) | O(b^d) |
| **Complete** | No | Yes |
| **Optimal** | No | Yes (with right algorithm) |
| **Speed** | Often faster | Can be slow |
| **Examples** | Min-conflicts, Genetic | DFS, BFS, A* |
| **Best for** | Large spaces, optimization | Small spaces, exact solutions |
| **In course** | N-queens, Knapsack | Frog-leap, N-puzzle |

### Q22: When to use which search algorithm?

**Use DFS when**:
- Deep solutions expected
- Memory limited
- Completeness not critical

**Use BFS when**:
- Shallow solutions
- Optimality required (unweighted)
- Memory available

**Use A*/IDA* when**:
- Have good heuristic
- Optimality required
- Memory limited (IDA*)

**Use local search when**:
- State space huge
- Good solution (not optimal) acceptable
- Fast answer needed

**In course**:
- **Frog-leap**: DFS (unique solution path)
- **N-puzzle**: IDA* (optimal solution, memory efficient)
- **N-queens**: Min-conflicts (huge space, any solution ok)
- **Knapsack**: Genetic (optimization, approximate ok)

### Q23: Compare classification algorithms.

| Algorithm | Speed | Accuracy | Interpretability | Assumptions |
|-----------|-------|----------|------------------|-------------|
| **K-NN** | Slow (query) | Good | Low | None |
| **Naive Bayes** | Fast | Good | Moderate | Independence |
| **Decision Tree** | Fast | Good | High | None |
| **SVM** | Moderate | High | Low | Margin-based |
| **Neural Net** | Slow (train) | High | Very low | None |

**In course**: K-NN, Naive Bayes, ID3 (Decision Tree).

---

## 6. Optimization & Heuristics

### Q24: What is a heuristic?
**A**: Rule of thumb or educated guess that guides search.

**Properties**:
- Not guaranteed correct
- Often effective in practice
- Reduces search space

**Examples in course**:
- **Manhattan distance**: Estimates puzzle solution distance
- **Value/weight ratio**: Guides knapsack repair
- **Information gain**: Chooses best split in ID3
- **Min-conflicts**: Chooses variable to reassign

### Q25: What are metaheuristics?
**A**: High-level strategies that guide search.

**Examples**:
- **Genetic algorithms**: Evolution-inspired
- **Simulated annealing**: Physics-inspired cooling
- **Tabu search**: Memory-based prohibition
- **Ant colony**: Swarm intelligence

**In course**: Genetic algorithm (knapsack).

**Characteristics**:
- Problem-independent frameworks
- Often stochastic (random)
- Trade optimality for speed

### Q26: What is the exploration vs. exploitation tradeoff?
**A**: Balance between trying new areas (exploration) vs. refining known good areas (exploitation).

**Too much exploration**: Wasting time on bad areas.
**Too much exploitation**: Miss better solutions elsewhere.

**In course**:
- **Genetic algorithm**: Mutation (explore) vs. selection (exploit)
- **Min-conflicts**: Random restart (explore) vs. hill climbing (exploit)

---

## 7. Complexity & Performance

### Q27: What is the curse of dimensionality?
**A**: Many algorithms degrade in high dimensions.

**Problems**:
1. **Distance becomes meaningless**: All points equidistant
2. **Data sparsity**: Need exponentially more data
3. **Computational cost**: Grows exponentially

**Affected algorithms**:
- **K-NN**: Distance-based, suffers badly
- **K-means**: Also distance-based
- **KD-tree**: Degrades to O(n) in high dimensions

**Mitigation**:
- Dimensionality reduction (PCA)
- Feature selection
- Distance metric learning

**In course**: Not a major issue (Iris: 4D, Votes: 16D).

### Q28: What is time complexity vs. space complexity?
**A**: 
**Time**: How many operations as input grows.
**Space**: How much memory as input grows.

**Trade-offs**:
- A*: Fast but memory-intensive
- IDA*: Slower but memory-efficient
- K-NN: Fast training (O(1)), slow query (O(n))
- Decision tree: Slow training (O(n log n)), fast query (O(depth))

### Q29: What is amortized analysis?
**A**: Average cost per operation over sequence.

**Example**: Dynamic array doubling.
- Most inserts: O(1)
- Occasional resize: O(n)
- Amortized: O(1) per insert

**In course**: Not explicitly used, but concept important.

---

## 8. Practical Considerations

### Q30: What is reproducibility?
**A**: Getting same results when re-running experiments.

**Requirements**:
- Fixed random seeds
- Same data ordering
- Same hyperparameters
- Same hardware (sometimes)

**In course**: Seeds set in all implementations:
- `random.seed(123)`, `np.random.seed(2003)`
- Ensures same train/test splits and CV folds

### Q31: What are hyperparameters?
**A**: Parameters set before training, not learned from data.

**Examples in course**:
- **K-NN**: k (number of neighbors)
- **ID3**: max_depth, min_samples, min_gain, alpha (chi-squared)
- **Naive Bayes**: alpha (Laplace smoothing)
- **K-means**: k (number of clusters)
- **Genetic**: population size, mutation rate, elite count

**Tuning**: Cross-validation to find best values.

### Q32: What is feature engineering?
**A**: Creating/selecting/transforming features to improve model.

**Techniques**:
- **Scaling**: Normalization (K-NN, K-means)
- **Encoding**: One-hot for categorical
- **Combination**: Polynomial features
- **Selection**: Remove irrelevant features
- **Extraction**: PCA, autoencoders

**In course**:
- **K-NN**: Min-max normalization
- **Naive Bayes**: Class-conditional imputation for missing values

---

## 9. Quick Definitions

### Q33: Define key terms.

**Entropy**: Measure of uncertainty/impurity.
- Formula: $H(S) = -\sum_i p_i \log_2(p_i)$
- Used in: ID3

**Information Gain**: Reduction in entropy from splitting.
- Formula: $IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$
- Used in: ID3

**Admissible heuristic**: Never overestimates true cost.
- Used in: IDA*

**Laplace smoothing**: Add pseudocounts to avoid zero probabilities.
- Used in: Naive Bayes

**Elitism**: Preserve best solutions across generations.
- Used in: Genetic algorithm

**Stratification**: Maintain class proportions in sampling.
- Used in: All supervised learning

**Alpha-beta pruning**: Eliminate branches that can't affect minimax decision.
- Used in: Tic-tac-toe

---

## 10. Summary of Course Algorithms

### Search Algorithms
| Algorithm | Type | Complete | Optimal | Memory |
|-----------|------|----------|---------|--------|
| **DFS** | Uninformed | No | No | O(d) |
| **IDA*** | Informed | Yes | Yes | O(d) |
| **Min-conflicts** | Local | No | No | O(1) |
| **Genetic** | Metaheuristic | No | No | O(population) |
| **Minimax** | Game tree | Yes | Yes | O(d) |

### Machine Learning Algorithms
| Algorithm | Type | Training | Query | Assumptions |
|-----------|------|----------|-------|-------------|
| **K-NN** | Instance-based | O(1) | O(n) | None |
| **Naive Bayes** | Probabilistic | O(n) | O(d) | Independence |
| **ID3** | Tree | O(n log n) | O(depth) | None |
| **K-means** | Clustering | O(ikn) | N/A | Spherical clusters |

---

## Final Defense Tips

1. **Know code locations**: Be able to point to exact line numbers.
2. **Understand alternatives**: Why this algorithm vs. others?
3. **Explain trade-offs**: Time, space, accuracy, assumptions.
4. **Define terms precisely**: Entropy, admissibility, overfitting, etc.
5. **Connect theory to code**: How is concept X implemented at line Y?
6. **Be ready for "what if"**: What if we changed parameter X?
7. **Complexity analysis**: Know Big-O for all algorithms.
8. **Evaluation metrics**: Accuracy, how it's calculated, when misleading.

**Remember**: Expect questions on ALL tasks, not just your chosen one!
