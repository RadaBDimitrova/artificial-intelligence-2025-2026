# Defense Documentation: K-Means Clustering (kmeans.py)

## Problem Overview
**Task**: Implement K-means and K-means++ clustering algorithms with multiple evaluation metrics.

**Algorithm**: K-means (random initialization + random restart), K-means++ (smart initialization)

**Metrics**: WCSS (Within-Cluster Sum of Squares), Silhouette Score, Davies-Bouldin Index

---

## Code Structure Explanation

### 1. Data Loading (Lines 14-21)
```python
def load_data(filename):
    points = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                x, y = map(float, line.split())
                points.append((x, y))
    return points
```
**What it does**: Reads 2D points from text file.
**Format**: Each line has "x y" coordinates.

### 2. Distance Function (Lines 24-25)
```python
def dist(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
```
**What it does**: Euclidean distance between two 2D points.
**Formula**: $d(p, q) = \sqrt{(p_x - q_x)^2 + (p_y - q_y)^2}$
**Where used**: Everywhere - assignment, centroid updates, metrics.

### 3. Cluster Building (Lines 28-32)
```python
def build_clusters(points, labels, k):
    clusters = [[] for _ in range(k)]
    for p, l in zip(points, labels):
        clusters[l].append(p)
    return clusters
```
**What it does**: Groups points by cluster assignment.
**Why**: Convenient for metric calculations and centroid updates.

### 4. K-Means Algorithm (Lines 35-63)

#### Signature (Line 35)
```python
def kmeans(points, k, centroids, max_iter=100):
```
**Parameters**:
- `points`: List of (x, y) tuples
- `k`: Number of clusters
- `centroids`: Initial centroid positions
- `max_iter`: Maximum iterations (default 100)

#### Assignment Step (Lines 40-46)
```python
for i, p in enumerate(points):
    dists = [dist(p, c) for c in centroids]
    label = dists.index(min(dists))
    if labels[i] != label:
        labels[i] = label
        changed = True
```
**What it does**: Assigns each point to nearest centroid.
**Location**: Inner loop, line 40-46.
**Convergence tracking**: `changed` flag detects when no reassignments occur.

#### Update Step (Lines 48-58)
```python
new_centroids = []
clusters = build_clusters(points, labels, k)
for cluster in clusters:
    if cluster:
        x = sum(p[0] for p in cluster) / len(cluster)
        y = sum(p[1] for p in cluster) / len(cluster)
        new_centroids.append((x, y))
    else:
        farthest = max(points, key=lambda p: min(dist(p, c) for c in new_centroids))
        new_centroids.append(farthest)
```
**What it does**:
1. Calculate mean position of points in each cluster
2. If cluster empty, assign farthest point from existing centroids

**Why handle empty clusters**: Prevents division by zero; ensures k clusters.

#### Convergence (Lines 60-62)
```python
centroids = new_centroids
if not changed:
    break
```
**What it does**: Stops when no point changes cluster.
**Guarantee**: Always converges (WCSS decreases monotonically).

### 5. K-Means++ Initialization (Lines 66-86)

#### Algorithm (Lines 67-85)
```python
def kmeans_plus_plus(points, k):
    centroids = [random.choice(points)]
    
    while len(centroids) < k:
        dists_sq = []
        for p in points:
            d = min(dist(p, c) for c in centroids)
            dists_sq.append(d * d)
        total = sum(dists_sq)
        r = random.uniform(0, total)
        
        # roulette wheel selection
        running_total = 0
        for p, w in zip(points, dists_sq):
            running_total += w
            if running_total >= r:
                centroids.append(p)
                break
    return centroids
```

**What it does**: Selects initial centroids smartly.

**Steps**:
1. **Line 67**: Choose first centroid randomly
2. **Lines 69-74**: For each point, calculate $D(x)^2$ (squared distance to nearest centroid)
3. **Lines 75-76**: Total sum for normalization
4. **Lines 78-84**: Roulette wheel selection - probability proportional to $D(x)^2$

**Why better than random**: Spreads centroids out, faster convergence, better results.

**Where**: Line 74 calculates squared distances.
**Where**: Lines 78-84 implement weighted random selection.

### 6. Evaluation Metrics

#### WCSS - Within-Cluster Sum of Squares (Lines 89-93)
```python
def wcss(points, labels, centroids):
    total = 0
    for p, l in zip(points, labels):
        total += dist(p, centroids[l]) ** 2
    return total
```
**Formula**: $WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$

**What it measures**: Compactness - how tight clusters are.

**Interpretation**: 
- Lower = better (tighter clusters)
- Decreases with more clusters
- Optimized by k-means directly

**Where**: Line 147 (random restart), line 155 (k-means++)

#### Silhouette Score (Lines 96-111)
```python
def silhouette(points, labels, k):
    n = len(points)
    score = 0
    for i in range(n):
        cluster = [points[j] for j in range(n) if labels[j] == labels[i] and j != i]
        other_cluster_groups = [...]
        
        a = sum(dist(points[i], p) for p in cluster) / len(cluster)
        b = min(sum(dist(points[i], p) for p in cluster) / len(cluster) 
                for cluster in other_cluster_groups)
        
        score += (b - a) / max(a, b)
    return score / n
```

**Formula**: For each point: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
- $a(i)$: Average distance to points in same cluster
- $b(i)$: Average distance to points in nearest other cluster

**Range**: [-1, 1]
- 1: Perfect (far from other clusters, close to own)
- 0: On boundary
- -1: Misclassified

**Interpretation**: Higher = better separation.

**Where**: 
- Line 104: Calculates $a(i)$ (intra-cluster distance)
- Line 105: Calculates $b(i)$ (nearest other cluster distance)
- Line 107: Combines into silhouette coefficient

#### Davies-Bouldin Index (Lines 114-131)
```python
def davies_bouldin(points, labels, centroids, k):
    clusters = build_clusters(points, labels, k)
    cluster_dists = []
    for i in range(k):
        if clusters[i]:
            avg_dist = sum(dist(p, centroids[i]) for p in clusters[i]) / len(clusters[i])
        else:
            avg_dist = 0
        cluster_dists.append(avg_dist)

    db_index = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                centroid_dist = dist(centroids[i], centroids[j])
                if centroid_dist != 0:
                    ratio = (cluster_dists[i] + cluster_dists[j]) / centroid_dist
                    max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / k
```

**Formula**: $DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{S_i + S_j}{d(c_i, c_j)}$
- $S_i$: Average distance within cluster i
- $d(c_i, c_j)$: Distance between centroids

**Interpretation**: 
- Lower = better (compact clusters, well-separated)
- 0 = ideal (impossible in practice)

**Where**: 
- Lines 117-121: Calculate within-cluster scatter $S_i$
- Lines 123-130: Find worst pair for each cluster, average

### 7. Random Restart K-Means (Lines 134-151)
```python
def random_restart_kmeans(points, k, metric, restarts=50):
    optimal_value = None
    optimal_labels = None
    optimal_centroids = None

    for _ in range(restarts):
        centroids = random.sample(points, k)
        labels, centroids = kmeans(points, k, centroids)
        
        if metric == 1:
            value = wcss(points, labels, centroids)
            changed = optimal_value is None or value < optimal_value
        elif metric == 2:
            value = silhouette(points, labels, k)
            changed = optimal_value is None or value > optimal_value
        else:
            value = davies_bouldin(points, labels, centroids, k)
            changed = optimal_value is None or value < optimal_value
            
        if changed:
            optimal_value = value
            optimal_labels = labels
            optimal_centroids = centroids

    return optimal_labels, optimal_centroids, optimal_value
```

**What it does**: Runs k-means 50 times with random initializations, keeps best.

**Why**: K-means sensitive to initialization; can get stuck in local optima.

**Metric-specific optimization**:
- **WCSS**: Minimize (line 145)
- **Silhouette**: Maximize (line 148)
- **Davies-Bouldin**: Minimize (line 150)

**Where**: Line 142 initializes randomly, line 143 runs k-means.

### 8. Main Execution (Lines 160-176)
```python
if __name__ == "__main__":
    filename = sys.argv[1]
    algorithm = sys.argv[2]  # "kmeans" or "kmeans++"
    metric = int(sys.argv[3])  # 1=WCSS, 2=Silhouette, 3=Davies-Bouldin
    k = int(sys.argv[4]
    points = load_data(filename)
    
    if algorithm == "kmeans":
        labels, centroids, value = random_restart_kmeans(points, k, metric)
    else:
        centroids = kmeans_plus_plus(points, k)
        labels, centroids = kmeans(points, k, centroids)
        # Calculate metric...
    
    save_and_plot(points, labels, centroids)
```

**Command line**: `python kmeans.py data.txt kmeans 1 3`
- data.txt: input file
- kmeans or kmeans++: algorithm
- 1, 2, or 3: metric
- 3: number of clusters

---

## Implementation Questions & Answers

### Q1: Where does k-means assign points to clusters?
**A**: Lines 40-46 in the `kmeans()` function:
```python
for i, p in enumerate(points):
    dists = [dist(p, c) for c in centroids]
    label = dists.index(min(dists))
```
Each point assigned to nearest centroid.

### Q2: How are centroids updated?
**A**: Lines 48-58. Calculate mean position:
```python
x = sum(p[0] for p in cluster) / len(cluster)
y = sum(p[1] for p in cluster) / len(cluster)
```
If cluster empty (line 54), assign farthest point.

### Q3: When does k-means converge?
**A**: Line 61-62:
```python
if not changed:
    break
```
Stops when no point changes cluster assignment in an iteration.

### Q4: How does k-means++ work?
**A**: Lines 66-86:
1. Choose first centroid randomly
2. For each remaining centroid:
   - Calculate $D(x)^2$ for each point (squared distance to nearest centroid)
   - Select next centroid with probability proportional to $D(x)^2$
3. Points far from existing centroids more likely to be chosen

**Where**: Lines 72-74 (distance calculation), 78-84 (weighted selection).

### Q5: What is random restart and where is it?
**A**: Lines 134-151. Runs k-means 50 times with different random initializations, keeps best result according to chosen metric.

**Why**: K-means finds local optimum; multiple restarts increase chance of finding global optimum.

### Q6: How do the three metrics differ?
**A**: 
- **WCSS** (lines 89-93): Measures compactness only
- **Silhouette** (lines 96-111): Measures both compactness and separation
- **Davies-Bouldin** (lines 114-131): Ratio of within-cluster to between-cluster distances

**Which to optimize**: 
- Lower WCSS/DB = better
- Higher Silhouette = better

### Q7: Where is the roulette wheel selection?
**A**: Lines 78-84 in `kmeans_plus_plus()`:
```python
running_total = 0
for p, w in zip(points, dists_sq):
    running_total += w
    if running_total >= r:
        centroids.append(p)
        break
```
Random value `r` chosen in [0, total], iterate until cumulative sum exceeds `r`.

### Q8: What happens if a cluster becomes empty?
**A**: Lines 54-56:
```python
else:
    farthest = max(points, key=lambda p: min(dist(p, c) for c in new_centroids))
    new_centroids.append(farthest)
```
Assigns the point farthest from all current centroids to the empty cluster.

---

## Theoretical Questions & Answers

### Q1: What is k-means clustering?
**A**: Unsupervised learning algorithm that partitions n points into k clusters.

**Goal**: Minimize within-cluster variance (WCSS).

**Algorithm**:
1. Initialize k centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as cluster means
4. Repeat 2-3 until convergence

### Q2: What is unsupervised learning?
**A**: Learning from unlabeled data - no target outputs.

**Goal**: Discover structure/patterns in data.

**Examples**: 
- Clustering (k-means, hierarchical)
- Dimensionality reduction (PCA)
- Anomaly detection

**vs. Supervised**: No labels, no "correct answer".

### Q3: Time and space complexity?
**A**: 
**Time**: $O(n \cdot k \cdot i \cdot d)$
- n = number of points
- k = number of clusters
- i = iterations until convergence
- d = dimensions (2 in this case)

**Space**: $O(n + k)$
- Store n points and k centroids

**Practical**: Usually converges quickly (i << n).

### Q4: Is k-means guaranteed to converge?
**A**: **Yes**, always converges to a local minimum.

**Proof**: 
- WCSS decreases (or stays same) each iteration
- WCSS bounded below by 0
- Finite number of possible assignments
- Must eventually stop changing

**However**: May converge to local (not global) minimum.

### Q5: Why use k-means++?
**A**: 
**Problem with random**: Centroids may start too close together.

**K-means++ solution**: Spread initial centroids out.

**Benefits**:
- Faster convergence (fewer iterations)
- Better final clustering (closer to global optimum)
- Theoretical guarantee: $O(\log k)$ approximation to optimal

**Trade-off**: Slightly more expensive initialization.

### Q6: How do you choose k?
**A**: Several methods:
1. **Elbow method**: Plot WCSS vs k, look for "elbow"
2. **Silhouette analysis**: Choose k with highest average silhouette
3. **Domain knowledge**: Know expected number of clusters
4. **Gap statistic**: Compare to random data

**This implementation**: k is input parameter.

### Q7: Limitations of k-means?
**A**: 
1. **Assumes spherical clusters**: Fails on elongated/irregular shapes
2. **Sensitive to outliers**: Mean calculation affected
3. **Requires k in advance**: Need to know/guess cluster count
4. **Local optima**: Random restart helps but doesn't guarantee global optimum
5. **Equal cluster sizes**: Tends to create similar-sized clusters

### Q8: What is the "k-means objective function"?
**A**: WCSS (Within-Cluster Sum of Squares):

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

K-means minimizes this by:
- **Assignment step**: Fix centroids, optimize assignments (line 40-46)
- **Update step**: Fix assignments, optimize centroids (line 48-58)

This is **coordinate descent** - alternately optimize different variables.

### Q9: Compare the three metrics.
**A**: 
| Metric | Formula | Range | Interpretation | Use Case |
|--------|---------|-------|----------------|----------|
| WCSS | $\sum ||x - \mu||^2$ | [0, ∞) | Lower better | Internal only |
| Silhouette | $\frac{b-a}{\max(a,b)}$ | [-1, 1] | Higher better | Compare k values |
| Davies-Bouldin | $\frac{1}{k}\sum \max \frac{S_i+S_j}{d_{ij}}$ | [0, ∞) | Lower better | Compare k values |

**WCSS**: Always decreases with more clusters (not good for choosing k).
**Silhouette**: Considers separation between clusters.
**Davies-Bouldin**: Ratio-based, less sensitive to scale.

### Q10: What is the difference between k-means and k-means++?
**A**: **Only initialization differs**.

| Aspect | K-means | K-means++ |
|--------|---------|-----------|
| Initialization | Random k points | Weighted probabilistic |
| Iterations | Same assignment/update | Identical to k-means |
| Convergence | Local optimum | Better local optimum |
| Time | Fast init | Slower init |
| Result quality | Variable | More consistent |

**Implementation**: Lines 66-86 (k-means++), line 142 (k-means random).

### Q11: Can k-means handle non-spherical clusters?
**A**: **No**, poorly.

**Why**: Uses Euclidean distance and mean - assumes isotropic Gaussian clusters.

**Alternatives**:
- **DBSCAN**: Density-based, arbitrary shapes
- **Gaussian Mixture Models**: Covariance-based, elliptical clusters
- **Spectral clustering**: Graph-based, complex shapes

### Q12: What is the "curse of dimensionality" for k-means?
**A**: In high dimensions:
1. **Distance becomes less meaningful**: All points equidistant
2. **Sparse data**: Need exponentially more data
3. **Computational cost**: $O(n \cdot k \cdot i \cdot d)$ grows with d

**Solutions**: Dimensionality reduction (PCA), feature selection.

---

## Quick Reference: Code Locations

| Functionality | Lines |
|---------------|-------|
| Data loading | 14-21 |
| Euclidean distance | 24-25 |
| Cluster building | 28-32 |
| K-means main | 35-63 |
| Assignment step | 40-46 |
| Update step | 48-58 |
| Convergence check | 61-62 |
| K-means++ | 66-86 |
| WCSS metric | 89-93 |
| Silhouette metric | 96-111 |
| Davies-Bouldin metric | 114-131 |
| Random restart | 134-151 |
| Main execution | 160-176 |

---

## Performance Characteristics

**Convergence**: 
- Typical: 10-30 iterations
- Worst case: Exponential (rare)

**Quality**: 
- Random init: Variable
- K-means++: More consistent
- Random restart: Best (50 tries)

**Scalability**: 
- Good for moderate n, k
- Struggles with high dimensions
