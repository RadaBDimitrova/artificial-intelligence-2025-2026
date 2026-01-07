import csv
import math
import random
from collections import Counter

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


def min_max_normalize(data):
    n_features = len(data[0])
    transposed = list(zip(*data))
    min_vals = [min(feature_col) for feature_col in transposed]
    max_vals = [max(feature_col) for feature_col in transposed]
    
    normalized_data = []
    for sample in data:
        normalized_sample = []
        for i in range(n_features):
            if max_vals[i] - min_vals[i] != 0:
                normalized_val = (sample[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
            else:
                normalized_val = 0.0
            normalized_sample.append(normalized_val)
        normalized_data.append(normalized_sample)
    
    return normalized_data


def stratified_splits(features, labels, test_size=0.2):
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    train_indices = []
    test_indices = []
    
    for _, indices in class_indices.items():
        shuffled_indices = indices[:]
        random.shuffle(shuffled_indices)
        
        n_test = int(len(shuffled_indices) * test_size)
        
        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])
    
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_features, train_labels, test_features, test_labels


def euclidean_distance(x1, x2):
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)


def neighbors(train_features, train_labels, test_instance, k):
    distances = []
    for i in range(len(train_features)):
        dist = euclidean_distance(test_instance, train_features[i])
        distances.append((dist, train_labels[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(k)]
    return neighbors


def evaluate_neighbors_predictions(neighbors):
    class_votes = Counter(neighbors)
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]


def knn(train_features, train_labels, test_features, k):
    predictions = []
    for test_instance in test_features:
        neighbors_list = neighbors(train_features, train_labels, test_instance, k)
        prediction = evaluate_neighbors_predictions(neighbors_list)
        predictions.append(prediction)
    
    return predictions


def get_accuracy(predictions, actual):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            correct += 1
    return correct / len(predictions)


def k_fold_split(features, labels, n_folds=10):
    class_indices = {}
    for i, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
    
    for indices in class_indices.values():
        random.shuffle(indices)
    
    folds = [[] for _ in range(n_folds)]
    for _, indices in class_indices.items():
        fold_size = len(indices) // n_folds
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(indices)
            folds[i].extend(indices[start:end])
    
    result_folds = []
    for i in range(n_folds):
        val_indices = folds[i]
        train_indices = []
        for j in range(n_folds):
            if j != i:
                train_indices.extend(folds[j])
        
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        
        train_features_fold = [features[idx] for idx in train_indices]
        train_labels_fold = [labels[idx] for idx in train_indices]
        val_features_fold = [features[idx] for idx in val_indices]
        val_labels_fold = [labels[idx] for idx in val_indices]
        
        result_folds.append((train_features_fold, train_labels_fold, val_features_fold, val_labels_fold))
    
    return result_folds


def cross_validate(features, labels, k, n_folds=10):
    folds = k_fold_split(features, labels, n_folds)
    accuracies = []
    
    for train_feat, train_lab, val_feat, val_lab in folds:
        predictions = knn(train_feat, train_lab, val_feat, k)
        accuracy = get_accuracy(predictions, val_lab)
        accuracies.append(accuracy)
    
    return accuracies


def get_mean(values):
    return sum(values) / len(values)


def get_var(values):
    mean = get_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# KD-TREE

class Node:
    def __init__(self, point, label, split_dim, left=None, right=None):
        self.point = point
        self.label = label
        self.split_dim = split_dim
        self.left = left
        self.right = right


def build(features, labels, depth=0):
    if not features:
        return None
    
    k = len(features[0])
    split_dim = depth % k
    
    sorted_indices = sorted(range(len(features)), key=lambda i: features[i][split_dim])
    median_idx = len(sorted_indices) // 2
    median_original_idx = sorted_indices[median_idx]
    

    left_indices = sorted_indices[:median_idx]
    right_indices = sorted_indices[median_idx + 1:]
    
    left_features = [features[i] for i in left_indices]
    left_labels = [labels[i] for i in left_indices]
    right_features = [features[i] for i in right_indices]
    right_labels = [labels[i] for i in right_indices]
    

    return Node(
        point=features[median_original_idx],
        label=labels[median_original_idx],
        split_dim=split_dim,
        left=build(left_features, left_labels, depth + 1),
        right=build(right_features, right_labels, depth + 1)
    )


def kdtree_neighbors(root, target, k):
    if root is None:
        return []
    best = []
    
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
    
    search(root)
    return [label for _, label in best]


def knn_with_kdtree(train_features, train_labels, test_features, k):
    kdtree = build(train_features, train_labels)
    
    predictions = []
    for test_instance in test_features:
        neighbors_labels = kdtree_neighbors(kdtree, test_instance, k)
        prediction = evaluate_neighbors_predictions(neighbors_labels)
        predictions.append(prediction)
    
    return predictions


def accuracy_graph(train_features, train_labels, test_features, test_labels, k_values=None, use_kdtree=False):
    if k_values is None:
        k_values = list(range(1, 21))
    
    train_accuracies = []
    test_accuracies = []
    cv_accuracies_list = []
    
    knn_eval = knn_with_kdtree if use_kdtree else knn 
    for k_val in k_values:
        train_pred = knn_eval(train_features, train_labels, train_features, k_val)
        train_acc = get_accuracy(train_pred, train_labels)
        train_accuracies.append(train_acc)

        test_pred = knn_eval(train_features, train_labels, test_features, k_val)
        test_acc = get_accuracy(test_pred, test_labels)
        test_accuracies.append(test_acc)

        cv_accs = cross_validate(train_features, train_labels, k_val, n_folds=10)
        cv_avg = get_mean(cv_accs)
        cv_accuracies_list.append(cv_avg)
        print(f"k={k_val:2d}: Train={train_acc*100:5.2f}% | "f"Cross={cv_avg*100:5.2f}% | Test={test_acc*100:5.2f}%")
    
    return k_values, train_accuracies, cv_accuracies_list, test_accuracies

def scale(acc, min_acc=0.70, max_acc=1.00, height=40):
    return int(((acc - min_acc) / (max_acc - min_acc)) * height)

def print_k_chart(k_values, train_accs, cv_accs, test_accs):
    all_accs = train_accs + cv_accs + test_accs
    min_acc = max(0.70, min(all_accs) - 0.02)
    max_acc = min(1.00, max(all_accs) + 0.02)
    
    height = 40

    for i in range(height, -1, -1):
        line = f"{min_acc + (i/height)*(max_acc-min_acc):.2f} |"
        
        for j, _ in enumerate(k_values):
            train_h = scale(train_accs[j], min_acc, max_acc, height)
            cv_h = scale(cv_accs[j], min_acc, max_acc, height)
            test_h = scale(test_accs[j], min_acc, max_acc, height)
            
            char_map = {train_h: "L", cv_h: "C", test_h: "T"}
            line += char_map.get(i, " ")
        
        print(line)


def normal_knn():
    k = int(input("k: "))
    
    features, labels = load_data('iris.data')
    normalized_features = min_max_normalize(features)
    train_features, train_labels, test_features, test_labels = stratified_splits(normalized_features, labels, test_size=0.2)
    
    print("\n1. Train Set Accuracy:")
    train_predictions = knn(train_features, train_labels, train_features, k)
    train_accuracy = get_accuracy(train_predictions, train_labels)
    print(f"Accuracy: {train_accuracy * 100:.2f}%\n")
    
    print("2. 10-Fold Cross-Validation Results:\n")
    cv_accuracies = cross_validate(train_features, train_labels, k, n_folds=10)
    for i, acc in enumerate(cv_accuracies, 1):
        print(f"Accuracy Fold {i}: {acc * 100:.2f}%")
    
    avg_accuracy = get_mean(cv_accuracies)
    std_accuracy = get_var(cv_accuracies)
    
    print(f"\nAverage Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Standard Deviation: {std_accuracy * 100:.2f}%\n")
    
    print("3. Test Set Accuracy:")
    test_predictions = knn(train_features, train_labels, test_features, k)
    test_accuracy = get_accuracy(test_predictions, test_labels)
    print(f"Accuracy: {test_accuracy * 100:.2f}%")


def knn_kdtree():
    k = int(input("k: "))
    
    features, labels = load_data('iris.data')
    normalized_features = min_max_normalize(features)
    train_features, train_labels, test_features, test_labels = stratified_splits(normalized_features, labels, test_size=0.2)    
    test_pred_standard = knn(train_features, train_labels, test_features, k)
    acc_standard = get_accuracy(test_pred_standard, test_labels)
    
    test_pred_kdtree = knn_with_kdtree(train_features, train_labels, test_features, k)
    acc_kdtree = get_accuracy(test_pred_kdtree, test_labels)
    
    print(f"kNN: {acc_standard*100:.2f}%")
    print(f"KD-tree kNN: {acc_kdtree*100:.2f}%")
    
    k_values = list(range(1, 2*k + 1))
    k_vals, train_accs, cv_accs, test_accs = accuracy_graph( train_features, train_labels, test_features, test_labels, k_values)

    print_k_chart(k_vals, train_accs, cv_accs, test_accs)

    best_k_idx = test_accs.index(max(test_accs))
    best_k = k_values[best_k_idx]
    print(f"\nBest k: {best_k} (Test Accuracy: {test_accs[best_k_idx]*100:.2f}%)")


def main():
    random.seed(2003)
    mode = input("Mode: 0 for Standard, 1 for Full with KD-tree: ").strip()
    if mode == "1":
        knn_kdtree()
    else:
        normal_knn()

if __name__ == "__main__":
    main()
