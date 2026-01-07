import sys
import random
import math
import numpy as np
from plot_clusters import plot_data_and_centroids

METRICS = {
    1: "WCSS",
    2: "Silhouette",
    3: "Davies-Bouldin"
}

def load_data(filename):
    points = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                x, y = map(float, line.split())
                points.append((x, y))
    return points


def dist(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def build_clusters(points, labels, k):
    clusters = [[] for _ in range(k)]
    for p, l in zip(points, labels):
        clusters[l].append(p)
    return clusters


def kmeans(points, k, centroids, max_iter=100):
    n = len(points)
    labels = [0] * n

    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            dists = [dist(p, c) for c in centroids]
            label = dists.index(min(dists))
            if labels[i] != label:
                labels[i] = label
                changed = True
                
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
                
        centroids = new_centroids
        if not changed:
            break
        
    return labels, centroids


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


def wcss(points, labels, centroids):
    total = 0
    for p, l in zip(points, labels):
        total += dist(p, centroids[l]) ** 2
    return total


def silhouette(points, labels, k):
    n = len(points)
    score = 0
    for i in range(n):
        cluster = [points[j] for j in range(n) if labels[j] == labels[i] and j != i]
        other_cluster_groups = []
        for c in range(k):
            if c != labels[i]:
                other_cluster = [points[j] for j in range(n) if labels[j] == c]
                other_cluster_groups.append(other_cluster)
        
        a = sum(dist(points[i], p) for p in cluster) / len(cluster) if cluster else 0
        b = min(sum(dist(points[i], p) for p in cluster) / len(cluster) for cluster in other_cluster_groups if cluster) if other_cluster_groups else 0
        
        score += (b - a) / max(a, b)
        
    return score / n


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


def save_and_plot(points, labels, centroids):
    np.savetxt('temp_data.txt', points)
    np.savetxt('temp_centroids.txt', centroids)
    np.savetxt('temp_labels.txt', labels, fmt='%d')
    plot_data_and_centroids('temp_data.txt', 'temp_centroids.txt', 'temp_labels.txt')


if __name__ == "__main__":
    filename = sys.argv[1]
    algorithm = sys.argv[2]
    metric = int(sys.argv[3])
    k = int(sys.argv[4])
    points = load_data(filename)
    
    if algorithm == "kmeans":
        labels, centroids, value = random_restart_kmeans(points, k, metric)
    else:
        centroids = kmeans_plus_plus(points, k)
        labels, centroids = kmeans(points, k, centroids)
        if metric == 1:
            value = wcss(points, labels, centroids)
        elif metric == 2:
            value = silhouette(points, labels, k)
        else:
            value = davies_bouldin(points, labels, centroids, k)
    
    save_and_plot(points, labels, centroids)
