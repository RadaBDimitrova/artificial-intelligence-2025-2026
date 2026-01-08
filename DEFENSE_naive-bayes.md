# Defense Documentation: Naive Bayes Classifier (naive-bayes.py)

## Problem Overview
**Task**: Implement Naive Bayes classifier for Congressional voting records (binary classification: Democrat/Republican).

**Dataset**: House Votes 84 - 16 voting features (yes/no/missing)

**Algorithm**: Naive Bayes with Laplace smoothing, class-conditional imputation for missing values

---

## Code Structure Explanation

### 1. Data Loading (Lines 5-13)
```python
def load_data():
    columns = ['class', 'handicapped-infants', 'water-project-cost-sharing', ...]
    data = pd.read_csv('house-votes-84.data', names=columns, na_values='?')
    return data
```

**Dataset**: 435 samples, 16 yes/no/abstain votes, 1 class label (democrat/republican).

### 2. Missing Value Handling (Lines 15-31)
```python
def preprocess_data(data, mode):
    if mode == 0:
        df = df.fillna('a')  # Fill with 'a' (abstain)
    else:
        # Class-conditional imputation
        for col in df.columns[1:]:
            for class_label in df['class'].unique():
                mask = (df['class'] == class_label) & (df[col].isna())
                if mask.any():
                    mode_value = df[df['class'] == class_label][col].mode()
                    df.loc[mask, col] = mode_value[0]
```

**Two modes**:
- **Mode 0**: Simple - fill with 'a' (abstain)
- **Mode 1**: Class-conditional - fill with most common vote within that class

**Where**: Lines 20-29 implement class-conditional imputation.

**Why mode 1 better**: Democrats/Republicans vote differently; using class-specific mode preserves patterns.

### 3. Stratified Splitting (Lines 39-57)
```python
def stratified_train_test_split(features, labels, test_size=0.2):
    classes = labels.unique()
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
```

**What it does**: 80/20 split maintaining class balance.

### 4. Naive Bayes Classifier Class (Lines 59-117)

#### Initialization (Lines 60-67)
```python
class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_priors = {}
        self.feature_probabilites = {}
        self.classes = None
        self.all_feature_values = {}
        self.class_counts = {}
```

**Alpha**: Laplace smoothing parameter (default 0.1).

#### Training (Lines 69-97)
```python
def train(self, features, labels):
    self.classes = labels.unique()
    
    # Calculate class priors P(C)
    for cls in self.classes:
        class_mask = labels == cls
        class_count = class_mask.sum()
        self.class_priors[cls] = np.log(class_count / n_samples)
```

**Step 1: Class Priors** (lines 77-82):
$$P(C) = \frac{\text{count}(C)}{N}$$

Stored as log probabilities to avoid underflow.

**Step 2: Feature Probabilities** (lines 84-97):
```python
for cls in self.classes:
    features_cls = features[labels == cls]
    for feature in features.columns:
        feature_values = self.all_feature_values[feature]
        n_values = len(feature_values)
        
        for value in feature_values:
            count = (features_cls[feature] == value).sum()
            prob = np.log((count + self.alpha) / (n_cls_samples + self.alpha * n_values))
            self.feature_probabilites[cls][feature][value] = prob
```

**Laplace Smoothing**:
$$P(f_i = v | C) = \frac{\text{count}(f_i = v, C) + \alpha}{N_C + \alpha \cdot |V|}$$

**Where**: Line 95 implements smoothed probability.

**Why log**: $\log P(C | X) = \log P(C) + \sum \log P(X_i | C)$ avoids underflow.

#### Prediction (Lines 99-113)
```python
def predict(self, features):
    predictions = []
    for idx in features.index:
        class_scores = {}
        for cls in self.classes:
            score = self.class_priors[cls]
            for feature in features.columns:
                value = features.loc[idx, feature]
                if value in self.feature_probabilites[cls][feature]:
                    score += self.feature_probabilites[cls][feature][value]
                else:
                    # Unseen value: use smoothing
                    score += np.log(self.alpha / (self.class_counts[cls] + self.alpha * len(self.all_feature_values[feature])))
            
            class_scores[cls] = score
        predicted_class = max(class_scores, key=class_scores.get)
```

**Formula**: $\arg\max_c [ \log P(C=c) + \sum_{i} \log P(X_i | C=c) ]$

**Where**: 
- Line 104: Initialize with class prior
- Lines 105-110: Add feature log-probabilities
- Line 112: Choose class with highest score

### 5. Cross-Validation (Lines 119-129, 131-155)

#### K-Fold Split (Lines 119-129)
```python
def stratified_kfold_split(labels, n_splits=10):
    indices = np.arange(len(labels))
    classes = labels.unique()
    folds = [[] for _ in range(n_splits)]
    
    for cls in classes:
        cls_indices = indices[labels.values == cls]
        np.random.shuffle(cls_indices)
        for i, idx in enumerate(cls_indices):
            folds[i % n_splits].append(idx)
```

**What it does**: Distributes samples round-robin by class.

#### Evaluation (Lines 131-155)
```python
def evaluate_model(features_train, labels_train, features_test, labels_test):
    model = NaiveBayesClassifier(alpha=0.1)
    model.train(features_train, labels_train)
    
    train_accuracy = model.score(features_train, labels_train) * 100
    
    cv_scores = []
    for train_idx, val_idx in stratified_kfold_split(labels_train, n_splits=10):
        fold_model = NaiveBayesClassifier(alpha=0.1)
        fold_model.train(features_fold_train, labels_fold_train)
        fold_accuracy = fold_model.score(features_fold_val, labels_fold_val) * 100
        cv_scores.append(fold_accuracy)
    
    test_accuracy = model.score(features_test, labels_test) * 100
```

**What it does**: Train/CV/test evaluation with α=0.1.

### 6. Main Function (Lines 157-184)
```python
def main():
    mode = int(input().strip())  # 0 or 1
    data = load_data()
    data = data.sample(frac=1, random_state=123).reset_index(drop=True)
    df_processed = preprocess_data(data, mode)
    
    features = df_processed.drop('class', axis=1)
    labels = df_processed['class']
    
    np.random.seed(123)
    features_train, features_test, labels_train, labels_test = stratified_train_test_split(features, labels, test_size=0.2)
    
    np.random.seed(2003)
    train_acc, cv_scores, cv_mean, cv_var, test_acc = evaluate_model(...)
```

**Seeds**: 123 for split, 2003 for CV - ensures reproducibility.

---

## Implementation Questions & Answers

### Q1: Where is Bayes' theorem applied?
**A**: Lines 99-112 in `predict()`:
$$P(C|X) \propto P(C) \prod P(X_i|C)$$
In log space: $\log P(C|X) = \log P(C) + \sum \log P(X_i|C)$

### Q2: What is the "naive" assumption?
**A**: **Feature independence given class**.
$$P(X_1, X_2, ..., X_n | C) = \prod P(X_i | C)$$
Voting issues are NOT truly independent, but assumption simplifies computation and works well in practice.

### Q3: Where is Laplace smoothing implemented?
**A**: Line 95:
```python
prob = np.log((count + self.alpha) / (n_cls_samples + self.alpha * n_values))
```
Adds α (0.1) to numerator, α × |values| to denominator.

### Q4: Why use logarithms?
**A**: **Numerical stability**.
- Probabilities are very small (e.g., 0.0001^16 = 10^-64)
- Multiplying many small numbers → underflow
- Logs convert multiplication to addition: $\log(ab) = \log a + \log b$

**Where**: Line 81 (priors), line 95 (features), line 106 (prediction).

### Q5: How are unseen feature values handled?
**A**: Lines 108-110:
```python
if value in self.feature_probabilites[cls][feature]:
    score += self.feature_probabilites[cls][feature][value]
else:
    score += np.log(self.alpha / (class_count + self.alpha * n_values))
```
Uses Laplace smoothing formula even if value never seen in training.

### Q6: What is class-conditional imputation?
**A**: Lines 20-29. Instead of filling all missing values with same value, fill based on class:
- Democrat with missing vote → fill with most common Democrat vote
- Republican with missing vote → fill with most common Republican vote

**Better than**: Global mode (ignores class patterns).

### Q7: Where is the model trained?
**A**: `train()` method, lines 69-97:
1. Calculate class priors (lines 77-82)
2. For each class, feature, and value: calculate $P(feature=value | class)$ (lines 84-97)

### Q8: How many parameters does the model have?
**A**: 
- **Class priors**: 2 (democrat, republican)
- **Feature probabilities**: 16 features × 3 values (y/n/a) × 2 classes = 96
- **Total**: ~98 parameters

Compare to logistic regression: 16 weights + 1 bias = 17 parameters.

---

## Theoretical Questions & Answers

### Q1: What is Naive Bayes?
**A**: Probabilistic classifier based on Bayes' theorem with naive independence assumption.

**Bayes' Theorem**:
$$P(C|X) = \frac{P(X|C) P(C)}{P(X)}$$

**Naive assumption**: Features independent given class:
$$P(X|C) = \prod_{i=1}^n P(X_i|C)$$

**Decision rule**:
$$\hat{y} = \arg\max_c P(C=c) \prod_{i=1}^n P(X_i=x_i | C=c)$$

### Q2: What is Laplace smoothing?
**A**: Technique to handle zero probabilities.

**Problem**: If feature value never seen in training for a class, $P(X_i = v | C) = 0$, entire product becomes 0.

**Solution**: Add α (pseudocount) to all counts:
$$P(X_i = v | C) = \frac{\text{count}(X_i = v, C) + \alpha}{N_C + \alpha \cdot |V|}$$

**α = 0**: No smoothing (fails on unseen values).
**α = 1**: Add-one smoothing.
**α = 0.1**: Weaker smoothing (this implementation).

**Where**: Line 95.

### Q3: Why is the independence assumption "naive"?
**A**: Real-world features are often correlated.

**Example**: Votes on related issues (environment bills) likely correlated.

**Despite being wrong**: Naive Bayes works surprisingly well!
- Fast training/prediction
- Works with small datasets
- Robust to irrelevant features

### Q4: Time and space complexity?
**A**: 
**Training**: $O(n \cdot d)$
- n samples, d features
- Count occurrences for each feature-value-class combination

**Prediction**: $O(d \cdot c)$ per sample
- d features, c classes
- Lookup d probabilities for each class

**Space**: $O(d \cdot v \cdot c)$
- Store probability for each feature, value, class combination
- v = values per feature

**This dataset**: d=16, v=3, c=2 → ~96 probabilities.

### Q5: Compare Naive Bayes to other classifiers.

| Aspect | Naive Bayes | Logistic Regression | Decision Tree | K-NN |
|--------|-------------|---------------------|---------------|------|
| Training | O(nd) | O(nd × iter) | O(n log n) | O(1) |
| Prediction | O(d) | O(d) | O(depth) | O(n) |
| Assumptions | Independence | Linear boundary | None | None |
| Interpretability | Moderate | High | High | Low |
| Handles missing | Yes (smoothing) | No (impute) | Yes (split) | No (impute) |

### Q6: When does Naive Bayes work well?
**A**: 
**Good for**:
- Text classification (spam detection)
- Categorical features
- Small datasets
- Real-time prediction (fast)
- Features relatively independent

**Poor for**:
- Highly correlated features
- Continuous features (needs discretization or Gaussian NB)
- Complex decision boundaries

### Q7: What is supervised learning?
**A**: Learning from labeled examples.

**Components**:
- Training data: (features, labels)
- Goal: Learn $f: X \rightarrow Y$
- Evaluation: Test on unseen data

**Naive Bayes**: Supervised - learns from (votes, party) pairs.

### Q8: Explain overfitting vs underfitting in Naive Bayes.
**A**: 
**Overfitting**: Rare, but can occur with:
- Too many features (curse of dimensionality)
- Very small training set
- No smoothing (α=0) → memorizes training data

**Underfitting**: 
- Independence assumption too strong
- Important feature correlations ignored

**This implementation**: α=0.1 prevents overfitting via smoothing.

### Q9: What is stratified sampling and why use it?
**A**: Sampling that preserves class proportions.

**Example**: 60% Democrat, 40% Republican.
- **Stratified**: Train and test both 60/40
- **Random**: Could be 70/30 in train, 45/55 in test

**Benefit**: More reliable evaluation, especially with imbalanced classes.

**Where**: Lines 39-57 (train/test), lines 119-129 (k-fold).

### Q10: Compare mode 0 vs mode 1 for missing values.
**A**: 
**Mode 0** (line 17): Fill all missing with 'a' (abstain).
- Simple, fast
- Ignores class patterns

**Mode 1** (lines 20-29): Fill with most common value within class.
- More sophisticated
- Preserves Democrat/Republican voting patterns
- Likely higher accuracy

**Example**: Missing vote on environmental bill.
- Mode 0: 'a' (abstain)
- Mode 1: 'y' if Democrat (typically vote yes), 'n' if Republican

### Q11: What is class prior?
**A**: $P(C)$ - probability of class before seeing features.

**Calculation**: $P(C=democrat) = \frac{\text{# democrats}}{\text{total samples}}$

**Where**: Lines 77-82:
```python
self.class_priors[cls] = np.log(class_count / n_samples)
```

**Example**: 267 democrats, 168 republicans → P(dem) ≈ 0.61.

### Q12: How does Naive Bayes handle imbalanced classes?
**A**: Naturally via class priors.

**Example**: 90% class A, 10% class B.
- Class prior: P(A) = 0.9, P(B) = 0.1
- B needs stronger evidence to be predicted
- Correctly reflects dataset distribution

**Alternative**: Can adjust priors manually for cost-sensitive learning.

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Data loading | 5-13 |
| Missing value handling | 15-31 |
| Stratified train/test split | 39-57 |
| NB classifier init | 60-67 |
| Training (priors) | 77-82 |
| Training (features) | 84-97 |
| Prediction | 99-113 |
| K-fold split | 119-129 |
| Evaluation | 131-155 |
| Main | 157-184 |

---

## Performance Characteristics

**Training**: Very fast - just counting.

**Prediction**: Very fast - table lookups and addition.

**Accuracy** (typical on voting data): 90-95%.

**Memory**: Small - stores probabilities only.

**Robustness**: Good with small data, handles missing values naturally.
