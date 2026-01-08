# Defense Documentation: ID3 Decision Tree (id3.py)

## Problem Overview
**Task**: Implement ID3 decision tree classifier for breast cancer classification with entropy-based splitting, pre-pruning, and post-pruning (chi-squared test).

**Algorithm**: ID3 with Information Gain, pre-pruning (max depth, min samples, min gain), chi-squared post-pruning

**Dataset**: Breast cancer data (binary classification)

---

## Code Structure Explanation

### 1. Data Loading (Lines 5-7)
```python
def load_data():
    cols = ["Class", "age", "menopause", ...]
    return pd.read_csv("breast-cancer.data", header=None, names=cols)
```
**What it does**: Loads CSV data and assigns column names.
**Where**: Entry point for data pipeline.

### 2. Missing Value Handling (Lines 10-14)
```python
def prep_missing(df):
    for col in df.columns:
        if (df[col] == "?").any():
            # Fill with most common value
```
**What it does**: Replaces '?' with mode (most frequent value).
**Why**: ID3 cannot handle missing values directly; imputation maintains data.

### 3. Stratified Splitting (Lines 17-27)
```python
def stratified_train_test_split(df, target, test_size=0.2):
```
**What it does**: Splits data maintaining class distribution in train/test.
**Why**: Prevents bias from imbalanced splits; ensures representative samples.
**Where**: Line 192 in main() for 80/20 split.

### 4. K-Fold Cross-Validation (Lines 30-40)
```python
def stratified_k_fold(df, target, k=10):
```
**What it does**: Creates 10 stratified folds for cross-validation.
**Why**: Robust evaluation; reduces variance in performance estimation.

### 5. Entropy Calculation (Lines 47-51)
```python
def entropy(labels):
    c = Counter(labels.values)
    total = len(labels)
    return -sum((val/total) * math.log2(val/total) for val in c.values())
```
**What it does**: Computes entropy $H(S) = -\sum_{i} p_i \log_2(p_i)$.
**Where**: Line 48-50, used in information gain calculation.
**Why**: Measures impurity/uncertainty in dataset.

**Example**: 
- Pure set (all one class): entropy = 0
- 50/50 split: entropy = 1 (maximum for binary)

### 6. Information Gain (Lines 54-64)
```python
def information_gain(features, labels, attr):
    base = entropy(labels)
    weighted_entropy = sum((len(subset)/len(labels)) * entropy(subset) 
                           for each value)
    return base - weighted_entropy
```
**What it does**: Calculates $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$.
**Where**: Line 100 - chooses attribute with max gain.
**Why**: Quantifies how much an attribute reduces uncertainty.

### 7. Node Class (Lines 66-71)
```python
class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # splitting attribute
        self.label = label          # class label (if leaf)
        self.children = {}          # child nodes per value
        self.class_counts = None    # class distribution
```
**What it does**: Represents tree node (internal or leaf).
**Structure**:
- Internal node: has `attribute` and `children`
- Leaf node: has `label`

### 8. ID3 Class (Lines 74-119)

#### Initialization (Lines 75-79)
```python
def __init__(self, max_depth=None, min_samples=1, min_gain=0):
```
**Parameters**:
- `max_depth`: Maximum tree depth (pre-pruning)
- `min_samples`: Minimum samples to split (pre-pruning)
- `min_gain`: Minimum info gain to split (pre-pruning)

#### Build Method (Lines 81-85)
```python
def build(self, features, labels, depth=0):
```
**What it does**: Initializes root and calls recursive builder.
**Where**: Line 205 in main().

#### Recursive Builder (Lines 87-110)
**Location**: `_build_recursive` method

**Stopping Conditions** (pre-pruning):
1. **Line 94**: Pure node (all same class) → create leaf
2. **Line 97**: No features left OR too few samples OR max depth → leaf
3. **Line 104**: Best gain below threshold → leaf

**Splitting Logic** (Lines 100-109):
```python
gains = {a: information_gain(features, labels, a) for a in features.columns}
best = max(gains, key=gains.get)
node.attribute = best

for val in features[best].unique():
    subset_features = features[features[best] == val].drop(columns=[best])
    subset_labels = labels[features[best] == val]
    node.children[val] = self._build_recursive(subset_features, subset_labels, depth + 1)
```
**What it does**:
1. Calculate info gain for all attributes
2. Choose attribute with maximum gain
3. Create child for each attribute value
4. Recursively build subtrees

### 9. Prediction (Lines 112-119)
```python
def _predict_sample(self, node, sample):
    if node.label is not None:
        return node.label
    if sample[node.attribute] in node.children:
        return self._predict_sample(node.children[sample[node.attribute]], sample)
    return max(node.class_counts, key=node.class_counts.get)
```
**What it does**:
1. If leaf, return label
2. Else, traverse to child matching sample's attribute value
3. If value unseen, return majority class at node

**Where**: Called by `predict()` on line 121.

### 10. Chi-Squared Post-Pruning (Lines 122-162)
```python
def chi_squared_prune(node, features, labels, alpha=0.01):
```
**What it does**: Prunes tree by converting nodes to leaves if split isn't statistically significant.

**Algorithm** (Lines 125-151):
1. Build contingency table: attribute values × class labels
2. Calculate expected frequencies: $E_{ij} = \frac{row_i \times col_j}{total}$
3. Compute chi-squared: $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$
4. Compare to critical value at significance level α
5. If $\chi^2 <$ critical value → merge children into leaf

**Where**: 
- Line 144: Chi-squared calculation
- Line 149: Significance test
- Line 151-154: Convert to leaf
- Line 156-158: Recursively prune children

**Why**: Prevents overfitting by removing splits that may be due to chance.

### 11. Cross-Validation (Lines 165-181)
```python
def cross_validate(df, target, params, use_chi2=True):
```
**What it does**:
1. Creates 10 stratified folds
2. For each fold: train on 9, validate on 1
3. Optionally applies chi-squared pruning
4. Returns accuracy scores

**Where**: Line 211 in main().

### 12. Main Function (Lines 184-232)

#### Mode Selection (Lines 186-195)
```python
mode = tokens[0]  # "0", "1", or "2"
use_pre = mode in ["0", "2"]   # pre-pruning
use_post = mode in ["1", "2"]  # post-pruning
```
**Modes**:
- **0**: Pre-pruning only
- **1**: Post-pruning only
- **2**: Both pre and post-pruning

#### Parameter Configuration (Lines 191-198)
```python
max_depth = 6 if (not flags or "N" in flags) and use_pre else None
min_samples = 5 if (not flags or "K" in flags) and use_pre else 1
min_gain = 0.01 if (not flags or "G" in flags) and use_pre else 0
use_chi2 = use_post and (not flags or "X" in flags)
```
**Flags**:
- **N**: Use max_depth=6
- **K**: Use min_samples=5
- **G**: Use min_gain=0.01
- **X**: Use chi-squared pruning

#### Training and Evaluation (Lines 200-214)
1. Preprocess data (line 200)
2. Split train/test 80/20 (line 201)
3. Build tree (line 206)
4. Apply chi-squared pruning if enabled (line 209)
5. Calculate train/test accuracy (line 211-212)
6. Perform 10-fold CV (line 215)

---

## Implementation Questions & Answers

### Q1: Where is the best attribute selected?
**A**: Line 100 in `_build_recursive()`:
```python
gains = {a: information_gain(features, labels, a) for a in features.columns}
best = max(gains, key=gains.get)
```

### Q2: How do you prevent overfitting?
**A**: Three mechanisms:
1. **Pre-pruning** (lines 75-79, 97-104): Stop early with max_depth, min_samples, min_gain
2. **Post-pruning** (lines 122-162): Chi-squared test removes insignificant splits
3. **Cross-validation** (lines 165-181): Evaluate generalization

### Q3: What happens when a test sample has an unseen attribute value?
**A**: Line 118 handles this:
```python
return max(node.class_counts, key=node.class_counts.get)
```
Returns the majority class at the current node.

### Q4: Where is entropy calculated?
**A**: Lines 47-51. Used in:
- Base entropy (line 55)
- Weighted entropy for each attribute value (line 59)
- Information gain (line 63)

### Q5: How does chi-squared pruning work?
**A**: Lines 122-162:
1. **Contingency table** (lines 127-135): Observed frequencies
2. **Expected frequencies** (line 144): Under independence assumption
3. **Chi-squared statistic** (line 145): Measures deviation
4. **Significance test** (line 150): Compare to critical value
5. **Prune** (lines 151-154): Convert to leaf if not significant

### Q6: What are the pre-pruning parameters?
**A**: 
- `max_depth=6` (line 192): Stop at depth 6
- `min_samples=5` (line 193): Need 5+ samples to split
- `min_gain=0.01` (line 194): Info gain must be > 0.01

### Q7: Where does stratification happen?
**A**: Two places:
1. **Train/test split** (line 201, function lines 17-27): Maintains class ratio
2. **K-fold CV** (line 215, function lines 30-40): Each fold balanced

### Q8: How is the tree recursively built?
**A**: Lines 107-109:
```python
for val in features[best].unique():
    subset = features[features[best] == val].drop(columns=[best])
    node.children[val] = self._build_recursive(subset, labels[subset], depth+1)
```
For each value of best attribute, filter data and recurse.

---

## Theoretical Questions & Answers

### Q1: What is the ID3 algorithm?
**A**: Iterative Dichotomiser 3 is a decision tree algorithm that:
- Uses **entropy** to measure impurity
- Selects splits by **information gain** (maximizes reduction in entropy)
- Builds tree top-down, greedily
- Only handles categorical features (in original form)

### Q2: Define entropy and information gain.
**A**: 
**Entropy**: $H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$
- Measures uncertainty/impurity
- 0 = pure, higher = more mixed

**Information Gain**: $IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$
- Expected reduction in entropy from splitting on attribute A
- Higher = more useful attribute

### Q3: What is overfitting and how do you avoid it?
**A**: 
**Overfitting**: Model learns training data too well, including noise; poor generalization.

**Signs**: High train accuracy, low test accuracy.

**Avoidance in ID3**:
1. **Pre-pruning**: Stop growing early (max_depth, min_samples, min_gain)
2. **Post-pruning**: Remove statistically insignificant branches (chi-squared)
3. **Cross-validation**: Detect overfitting during training
4. **More data**: Reduces noise impact

### Q4: What is underfitting?
**A**: Model too simple to capture data patterns; poor train AND test accuracy.

**Causes**: Too much pruning, insufficient features.

**Solution**: Reduce pruning constraints, add features, use more complex model.

### Q5: Compare pre-pruning vs post-pruning.
**A**: 
| Aspect | Pre-pruning | Post-pruning |
|--------|-------------|--------------|
| **When** | During tree building | After tree built |
| **Method** | Stop early | Build full, then prune |
| **Pros** | Faster, less memory | More accurate, principled |
| **Cons** | May underfit | Slower, more memory |
| **Implementation** | max_depth, min_samples | Chi-squared test |

### Q6: Why stratified sampling?
**A**: 
- **Problem**: Random split may create imbalanced train/test sets
- **Solution**: Stratification maintains class proportions
- **Benefit**: More reliable evaluation, especially with imbalanced data

**Example**: 70% class A, 30% class B
- Stratified: Both train and test have 70/30 split
- Random: Could be 80/20 in train, 50/50 in test

### Q7: What is chi-squared test doing?
**A**: Tests **null hypothesis**: "Attribute and class are independent"
- **Low χ²**: Split could be random → prune
- **High χ²**: Split is significant → keep

**Formula**: $\chi^2 = \sum \frac{(O - E)^2}{E}$
- O = observed frequency
- E = expected frequency under independence
- Degrees of freedom: $(rows-1) \times (cols-1)$

### Q8: What are alternatives to ID3?
**A**: 
1. **C4.5**: Handles continuous attributes, uses gain ratio (bias correction)
2. **CART**: Binary splits, Gini impurity, can do regression
3. **Random Forest**: Ensemble of trees, reduces overfitting
4. **Gradient Boosting**: Sequential trees, corrects previous errors

### Q9: Time and space complexity?
**A**: 
**Time**: 
- Training: $O(m \cdot n \cdot \log n)$ where m=features, n=samples
- Each level: check all features, compute gain
- Depth: $O(\log n)$ to $O(n)$

**Space**: 
- Tree: $O(n)$ in worst case (each sample a leaf)
- Typically: $O(\log n)$ for balanced tree

### Q10: Is ID3 optimal?
**A**: 
- **Greedy**: Chooses best split locally, not globally optimal
- **NP-complete**: Finding optimal decision tree is NP-complete
- **Practical**: Works well in practice despite being greedy

### Q11: How does ID3 handle continuous features?
**A**: Original ID3 doesn't. Extensions:
1. **Discretization**: Bin continuous values
2. **Binary splits**: Find threshold (like C4.5)
3. **This implementation**: Assumes categorical (breast cancer data is categorical)

### Q12: What is supervised learning?
**A**: Learning from labeled data (input-output pairs).
- **Input**: Features (attributes)
- **Output**: Labels (class)
- **Goal**: Learn mapping function
- **ID3**: Supervised - learns from (features, class) pairs

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Data loading | 5-7 |
| Missing value handling | 10-14 |
| Train/test split | 17-27 |
| K-fold CV setup | 30-40 |
| Entropy calculation | 47-51 |
| Information gain | 54-64 |
| Node class | 66-71 |
| ID3 initialization | 75-79 |
| Tree building (main) | 81-85 |
| Recursive builder | 87-110 |
| Prediction | 112-119 |
| Chi-squared pruning | 122-162 |
| Cross-validation | 165-181 |
| Main function | 184-232 |

---

## Performance Characteristics

**This Implementation**:
- **Complete**: Yes (always produces tree)
- **Deterministic**: Tie-breaking is random (line 101)
- **Overfitting prevention**: Pre + post pruning
- **Evaluation**: 10-fold CV for robust accuracy

**Typical Results**:
- Train accuracy: 95-100% (may indicate overfitting if too high)
- Test accuracy: 70-85% (breast cancer is challenging)
- CV accuracy: Best indicator of true performance
