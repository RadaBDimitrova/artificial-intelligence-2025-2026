# Defense Documentation: K-Nearest Neighbors (knn_iris.py)

## Problem Overview
**Task**: Implement K-NN classifier for Iris dataset with optional KD-tree optimization.

**Dataset**: Iris flowers - 4 features, 3 classes (setosa, versicolor, virginica)

**Algorithm**: K-NN (brute force), K-NN with KD-tree (optimized)

---

## Code Structure Explanation

### 1. Data Loading (Lines 6-18)
```python
def load_data(filename='iris.data'):
    features = []
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 5 and row[0]:
                feature_vector = [float(row[i]) for i in range(4)]
                features.append(feature_vector)
                labels.append(row[4])
    return features, labels
```

**What it does**: Loads CSV file with 4 features + 1 label per sample.

**Features**: Sepal length, sepal width, petal length, petal width.

### 2. Min-Max Normalization (Lines 21-38)
```python
def min_max_normalize(data):
    transposed = list(zip(*data))
    min_vals = [min(feature_col) for feature_col in transposed]
    max_vals = [max(feature_col) for feature_col in transposed]
    
    for sample in data:
        normalized_sample = []
        for i in range(n_features):
            normalized_val = (sample[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            normalized_sample.append(normalized_val)
```

**Formula**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

**Result**: All features scaled to [0, 1].

**Why needed**: K-NN is distance-based; features with larger scales dominate.

**Where**: Line 24-25 find min/max per feature.

### 3. Stratified Train/Test Split (Lines 41-67)
```python
def stratified_splits(features, labels, test_size=0.2):
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    for _, indices in class_indices.items():
        shuffled_indices = indices[:]
        random.shuffle(shuffled_indices)
        n_test = int(len(shuffled_indices) * test_size)
        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])
```

**What it does**: 80/20 split maintaining class proportions.

**Why stratified**: Iris has balanced classes; prevents accidental imbalance.

### 4. Euclidean Distance (Lines 70-73)
```python
def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)
```

**Formula**: $d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$

**Where used**: Everywhere - neighbor finding, KD-tree search.

### 5. K-NN Core Algorithm

#### Find Neighbors (Lines 76-84)
```python
def neighbors(train_features, train_labels, test_instance, k):
    distances = []
    for i in range(len(train_features)):
        dist = euclidean_distance(test_instance, train_features[i])
        distances.append((dist, train_labels[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(k)]
    return neighbors
```

**What it does**:
1. Calculate distance to all training points
2. Sort by distance
3. Return labels of k nearest neighbors

**Where**: Line 79 calculates distances, line 82 sorts, line 83 selects k nearest.

#### Make Prediction (Lines 87-90)
```python
def evaluate_neighbors_predictions(neighbors):
    class_votes = Counter(neighbors)
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]
```

**What it does**: Majority vote among k neighbors.

**Example**: k=5, neighbors = [A, A, B, A, C] → prediction = A.

#### Predict All (Lines 93-100)
```python
def knn(train_features, train_labels, test_features, k):
    predictions = []
    for test_instance in test_features:
        neighbors_list = neighbors(train_features, train_labels, test_instance, k)
        prediction = evaluate_neighbors_predictions(neighbors_list)
        predictions.append(prediction)
    return predictions
```

**What it does**: Predicts class for all test instances.

### 6. Cross-Validation

#### K-Fold Split (Lines 112-147)
```python
def k_fold_split(features, labels, n_folds=10):
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    folds = [[] for _ in range(n_folds)]
    for _, indices in class_indices.items():
        fold_size = len(indices) // n_folds
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(indices)
            folds[i].extend(indices[start:end])
```

**What it does**: Creates 10 stratified folds.

**Process**:
1. Group indices by class
2. Divide each class into 10 parts
3. Distribute across folds

**Where**: Lines 115-119 group by class, lines 124-128 split into folds.

#### Cross-Validate (Lines 150-160)
```python
def cross_validate(features, labels, k, n_folds=10):
    folds = k_fold_split(features, labels, n_folds)
    accuracies = []
    
    for train_feat, train_lab, val_feat, val_lab in folds:
        predictions = knn(train_feat, train_lab, val_feat, k)
        accuracy = get_accuracy(predictions, val_lab)
        accuracies.append(accuracy)
    
    return accuracies
```

**What it does**: Runs K-NN on each fold, returns 10 accuracy scores.

### 7. KD-Tree Implementation

#### Node Class (Lines 176-182)
```python
class Node:
    def __init__(self, point, label, split_dim, left=None, right=None):
        self.point = point        # Data point
        self.label = label        # Class label
        self.split_dim = split_dim  # Which dimension to split on
        self.left = left          # Left subtree
        self.right = right        # Right subtree
```

**Structure**: Binary tree, alternates splitting dimensions.

#### Build KD-Tree (Lines 185-211)
```python
def build(features, labels, depth=0):
    if not features:
        return None
    
    k = len(features[0])
    split_dim = depth % k  # Cycle through dimensions
    
    sorted_indices = sorted(range(len(features)), key=lambda i: features[i][split_dim])
    median_idx = len(sorted_indices) // 2
    median_original_idx = sorted_indices[median_idx]
    
    left_indices = sorted_indices[:median_idx]
    right_indices = sorted_indices[median_idx + 1:]
    
    return Node(
        point=features[median_original_idx],
        label=labels[median_original_idx],
        split_dim=split_dim,
        left=build(left_features, left_labels, depth + 1),
        right=build(right_features, right_labels, depth + 1)
    )
```

**What it does**:
1. Choose splitting dimension (cycle: 0, 1, 2, 3, 0, ...)
2. Find median along that dimension
3. Recursively build left and right subtrees

**Where**: Line 190 selects dimension, line 192-194 find median, line 196-210 recursive build.

**Time**: $O(n \log^2 n)$ to build.

#### KD-Tree Search (Lines 214-245)
```python
def kdtree_neighbors(root, target, k):
    best = []  # Max-heap of (negative_distance, label)
    
    def search(node):
        if node is None:
            return
        
        dist = euclidean_distance(target, node.point)
        if len(best) < k:
            best.append((-dist, node.label))
            best.sort(reverse=True)
        elif dist < -best[0][0]:
            best[0] = (-dist, node.label)
            best.sort(reverse=True)

        split_dim = node.split_dim
        diff = target[split_dim] - node.point[split_dim]
        if diff < 0:
            close, far = node.left, node.right
        else:
            close, far = node.right, node.left

        search(close)
        if len(best) < k or abs(diff) < -best[0][0]:
            search(far)
```

**What it does**:
1. Maintain heap of k nearest neighbors
2. Traverse tree, prioritizing closer subtrees
3. Prune subtrees that can't contain closer neighbors

**Pruning logic** (line 239): If distance to splitting plane > distance to kth neighbor, skip far subtree.

**Where**: Lines 222-229 update best neighbors, lines 231-235 decide traversal order, line 238-239 pruning.

**Time**: $O(k \log n)$ average, $O(k n)$ worst case.

### 8. Evaluation and Visualization

#### Accuracy Calculation (Lines 103-109)
```python
def get_accuracy(predictions, actual):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            correct += 1
    return correct / len(predictions)
```

#### Accuracy vs k Graph (Lines 260-282)
```python
def accuracy_graph(train_features, train_labels, test_features, test_labels, k_values=None, use_kdtree=False):
    knn_eval = knn_with_kdtree if use_kdtree else knn 
    for k_val in k_values:
        train_pred = knn_eval(train_features, train_labels, train_features, k_val)
        train_acc = get_accuracy(train_pred, train_labels)
        
        test_pred = knn_eval(train_features, train_labels, test_features, k_val)
        test_acc = get_accuracy(test_pred, test_labels)
        
        cv_accs = cross_validate(train_features, train_labels, k_val, n_folds=10)
        cv_avg = get_mean(cv_accs)
```

**What it does**: Evaluates K-NN for different k values (1-20), shows train/CV/test accuracy.

**Purpose**: Find optimal k value.

### 9. Main Functions

#### Standard K-NN (Lines 310-333)
```python
def normal_knn():
    k = int(input("k: "))
    features, labels = load_data('iris.data')
    normalized_features = min_max_normalize(features)
    train_features, train_labels, test_features, test_labels = stratified_splits(normalized_features, labels, test_size=0.2)
    
    train_predictions = knn(train_features, train_labels, train_features, k)
    cv_accuracies = cross_validate(train_features, train_labels, k, n_folds=10)
    test_predictions = knn(train_features, train_labels, test_features, k)
```

**What it does**: Standard K-NN with train/CV/test evaluation.

#### KD-Tree K-NN (Lines 336-357)
```python
def knn_kdtree():
    k = int(input("k: "))
    # Compare standard vs KD-tree
    test_pred_standard = knn(train_features, train_labels, test_features, k)
    test_pred_kdtree = knn_with_kdtree(train_features, train_labels, test_features, k)
    
    # Evaluate multiple k values
    k_values = list(range(1, 2*k + 1))
    accuracy_graph(train_features, train_labels, test_features, test_labels, k_values)
```

**What it does**: Compares standard vs KD-tree, finds best k.

---

## Implementation Questions & Answers

### Q1: Where does K-NN make predictions?
**A**: `evaluate_neighbors_predictions()` (lines 87-90):
```python
class_votes = Counter(neighbors)
sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
return sorted_votes[0][0]
```
Majority vote among k neighbors.

### Q2: How are distances calculated?
**A**: `euclidean_distance()` (lines 70-73):
```python
distance = sum((x1[i] - x2[i]) ** 2 for i in range(len(x1)))
return math.sqrt(distance)
```
Standard Euclidean distance in 4D space.

### Q3: Why normalize features?
**A**: Line 21-38. K-NN uses distance - features with larger scales (e.g., petal length 1-7cm) would dominate over smaller scales (e.g., petal width 0.1-2.5cm). Normalization puts all features on equal footing [0, 1].

### Q4: Where is the KD-tree built?
**A**: `build()` function (lines 185-211):
- Line 190: Choose split dimension (depth % 4)
- Line 192-194: Find median
- Line 196-210: Recursive build

### Q5: How does KD-tree search prune branches?
**A**: Line 238-239:
```python
if len(best) < k or abs(diff) < -best[0][0]:
    search(far)
```
Only search far subtree if:
- We don't have k neighbors yet, OR
- Distance to splitting plane < distance to kth neighbor

### Q6: What is the time complexity of brute force K-NN?
**A**: 
- **Training**: $O(1)$ (just store data)
- **Prediction**: $O(n \cdot d)$ per query where n=training samples, d=dimensions
  - Calculate n distances: $O(n \cdot d)$
  - Sort: $O(n \log n)$ but we only need k smallest: $O(n)$ with quickselect
- **Total for m queries**: $O(m \cdot n \cdot d)$

### Q7: What is the time complexity with KD-tree?
**A**: 
- **Building**: $O(d \cdot n \log^2 n)$
- **Query**: $O(k \log n)$ average, $O(k \cdot n)$ worst case
- **Total for m queries**: $O(d \cdot n \log^2 n + m \cdot k \log n)$

**Better when**: $m$ is large (many queries).

### Q8: How is cross-validation implemented?
**A**: Lines 112-160:
1. `k_fold_split()`: Divides data into 10 stratified folds
2. `cross_validate()`: For each fold, train on 9, validate on 1
3. Returns 10 accuracy scores

**Stratified**: Each fold maintains class proportions.

---

## Theoretical Questions & Answers

### Q1: What is K-Nearest Neighbors?
**A**: Instance-based (lazy) learning algorithm.

**Classification**:
1. Find k nearest training samples to test point
2. Majority vote determines class

**"Lazy"**: No training phase - stores data and computes at query time.

**Supervised**: Requires labeled training data.

### Q2: How do you choose k?
**A**: 
**Methods**:
1. **Cross-validation**: Try different k, choose best CV accuracy
2. **Domain knowledge**: Task-specific insights
3. **Rule of thumb**: $k = \sqrt{n}$

**Trade-off**:
- **Small k**: Low bias, high variance (overfitting)
- **Large k**: High bias, low variance (underfitting)
- **Optimal**: Balance (often k=3-7 for Iris)

**This implementation**: `accuracy_graph()` evaluates multiple k values.

### Q3: Why normalize features?
**A**: 
**Problem**: Features on different scales.
- Sepal length: 4.3-7.9 cm
- Petal width: 0.1-2.5 cm

**Without normalization**: Sepal length dominates distance calculation.

**Solution**: Min-max scaling to [0, 1]:
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Alternatives**: Z-score normalization, L2 normalization.

### Q4: What is a KD-tree?
**A**: K-dimensional tree - binary space partitioning structure.

**Construction**:
1. Choose dimension (cycle through)
2. Split at median along that dimension
3. Recursively build left/right subtrees

**Search**:
1. Traverse to leaf (following splits)
2. Backtrack, checking if other branches could have closer points
3. Prune branches geometrically impossible to contain closer points

**Benefit**: $O(\log n)$ average query vs $O(n)$ brute force.

### Q5: When does KD-tree degrade?
**A**: 
**High dimensions** (curse of dimensionality):
- Query time approaches $O(n)$ when $d \geq 20$
- Space partitioning becomes less effective
- Most points become similar distance

**Worst case**: All points equidistant → must check all.

**Iris**: 4 dimensions - KD-tree works well.

### Q6: What are alternatives to Euclidean distance?
**A**: 
1. **Manhattan** (L1): $\sum |x_i - y_i|$
2. **Minkowski**: $(\sum |x_i - y_i|^p)^{1/p}$
3. **Cosine**: $\frac{x \cdot y}{||x|| ||y||}$
4. **Hamming**: For categorical features

**This implementation**: Only Euclidean (lines 70-73).

### Q7: Is K-NN parametric or non-parametric?
**A**: **Non-parametric**.

**Parametric**: Assumes fixed functional form (e.g., linear, Gaussian).

**Non-parametric**: No fixed form; complexity grows with data.

**K-NN**: Stores all training data; model size = training set size.

### Q8: Compare K-NN to other classifiers.

| Aspect | K-NN | Decision Tree | Naive Bayes | SVM |
|--------|------|---------------|-------------|-----|
| Training time | O(1) | O(n log n) | O(n) | O(n²) to O(n³) |
| Query time | O(n) or O(log n) | O(depth) | O(d) | O(n_{sv}) |
| Interpretability | Moderate | High | High | Low |
| Overfitting | k controls | Pruning helps | Rare | Kernel/C control |

### Q9: What is stratified sampling?
**A**: Sampling that preserves class proportions.

**Example**: Iris has 50 of each class.
- **Stratified 80/20**: Train has 40 each, test has 10 each
- **Random 80/20**: Could be 35/45/40 in train, 15/5/10 in test

**Benefit**: More representative splits, especially with imbalanced data.

**Where**: Lines 41-67 (`stratified_splits`), lines 112-147 (`k_fold_split`).

### Q10: What is the curse of dimensionality?
**A**: Phenomenon where algorithms degrade in high dimensions.

**For K-NN**:
1. **Distance becomes meaningless**: All points equidistant
2. **Data sparsity**: Need exponentially more data
3. **KD-tree ineffective**: Can't prune branches

**Formula**: Volume of unit sphere in d dimensions shrinks relative to cube.

**Solution**: Dimensionality reduction (PCA), feature selection.

**Iris**: Only 4 dimensions - not affected.

### Q11: How does K-NN handle class imbalance?
**A**: **Majority vote can be biased**.

**Example**: 9 class A, 1 class B in training. k=5 → likely predicts A.

**Solutions**:
1. **Weighted voting**: Weight by inverse distance
2. **Adjust k**: Smaller k gives minority class better chance
3. **SMOTE**: Synthetic minority oversampling

**This implementation**: Simple majority vote (line 87-90).

### Q12: What is supervised learning?
**A**: Learning from labeled data.

**Components**:
- **Input**: Features (X)
- **Output**: Labels (y)
- **Goal**: Learn $f: X \rightarrow y$

**K-NN**: Supervised - learns from (features, class) pairs.

**vs. Unsupervised**: No labels (e.g., K-means clustering).

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Data loading | 6-18 |
| Normalization | 21-38 |
| Train/test split | 41-67 |
| Euclidean distance | 70-73 |
| Find k neighbors | 76-84 |
| Majority vote | 87-90 |
| K-NN prediction | 93-100 |
| Accuracy calculation | 103-109 |
| K-fold split | 112-147 |
| Cross-validation | 150-160 |
| KD-tree node | 176-182 |
| Build KD-tree | 185-211 |
| KD-tree search | 214-245 |
| KD-tree K-NN | 248-256 |
| Accuracy vs k | 260-282 |
| Standard K-NN main | 310-333 |
| KD-tree K-NN main | 336-357 |

---

## Performance Characteristics

**Brute Force K-NN**:
- Train: O(1)
- Query: O(n)
- Simple, accurate for small datasets

**KD-Tree K-NN**:
- Build: O(n log² n)
- Query: O(log n) average
- Better for large datasets, low dimensions

**Iris Performance** (typical):
- Accuracy: 95-98%
- Optimal k: 3-7
- KD-tree faster for n > 100
