import pandas as pd
import numpy as np
import math
from collections import Counter
from scipy.stats import chi2

def load_data():
    cols = [ "Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
    return pd.read_csv("breast-cancer.data", header=None, names=cols)


def prep_missing(df):
    for col in df.columns:
        if (df[col] == "?").any():
            for cls in df["Class"].unique():
                mode = df[(df[col] != "?") & (df["Class"] == cls)][col].mode()
                if not mode.empty:
                    df.loc[(df[col] == "?") & (df["Class"] == cls), col] = mode[0]
    return df


def stratified_train_test_split(df, target, test_size=0.2):
    train, test = [], []
    for cls in df[target].unique():
        part = df[df[target] == cls].sample(frac=1)
        cut = int(len(part) * (1 - test_size))
        train.append(part.iloc[:cut])
        test.append(part.iloc[cut:])

    train_set = pd.concat(train).sample(frac=1)
    test_set = pd.concat(test).sample(frac=1)
    return train_set, test_set


def stratified_k_fold(df, target, k=10):
    folds = [[] for _ in range(k)]

    for cls in df[target].unique():
        cls_df = df[df[target] == cls].sample(frac=1)
        splits = np.array_split(cls_df, k)
        for i in range(k):
            folds[i].append(splits[i])

    return [pd.concat(f).sample(frac=1) for f in folds]


def accuracy(labels_true, labels_pred):
    return np.mean(labels_true == labels_pred) * 100


def entropy(labels):
    c = Counter(labels.values)
    total = len(labels)
    return -sum((val/total) * math.log2(val/total) for val in c.values()) if total > 0 else 0


def information_gain(features, labels, attr):
    base = entropy(labels)
    values = features[attr].unique()
    weighted_entropy = 0

    for val in values:
        labels_sub = labels[features[attr] == val]
        weighted_entropy += (len(labels_sub)/len(labels)) * entropy(labels_sub)

    return base - weighted_entropy

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute
        self.label = label
        self.children = {}
        self.class_counts = None


class ID3:
    def __init__(self, max_depth=None, min_samples=1, min_gain=0):
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_gain = min_gain

    def build(self, features, labels, depth=0):
        if depth == 0:
            self.root = self._build_recursive(features, labels, 0)
            return self.root
        return self._build_recursive(features, labels, depth)

    def _build_recursive(self, features, labels, depth):
        node = Node()
        node.class_counts = Counter(labels.values)
        majority_class = node.class_counts.most_common(1)[0][0]

        if len(node.class_counts) == 1:
            return Node(label=majority_class)

        if features.empty or len(labels) < self.min_samples or (self.max_depth is not None and depth >= self.max_depth):
            return Node(label=majority_class)

        gains = {a: information_gain(features, labels, a) for a in features.columns}
        best = max(gains, key=gains.get)

        if gains[best] < self.min_gain:
            return Node(label=majority_class)
        node.attribute = best

        for val in features[best].unique():
            idx = features[best] == val
            node.children[val] = self._build_recursive(features[idx].drop(columns=[best]), labels[idx], depth + 1)
        return node

    def _predict_sample(self, node, sample):
        if node.label is not None:
            return node.label
        if sample[node.attribute] in node.children:
            return self._predict_sample(node.children[sample[node.attribute]], sample)
        return max(node.class_counts, key=node.class_counts.get)

    def predict(self, features):
        return features.apply(lambda r: self._predict_sample(self.root, r), axis=1)


def chi_squared_prune(node, features, labels, alpha=0.01):
    if node.label is not None or node.attribute is None:
        return

    contingency = []
    classes = list(set(labels))
    for val, child in node.children.items():
        row = []
        idx = features[node.attribute] == val
        for cls in classes:
            row.append(sum(labels[idx] == cls))
        contingency.append(row)

    contingency = np.array(contingency)
    if contingency.shape[0] < 2:
        return

    total = contingency.sum()
    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)

    expected = np.outer(row_sum, col_sum) / total
    chi2_stat = ((contingency - expected) ** 2 / expected).sum()

    degrees_of_freedom = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)

    if chi2_stat < critical_value:
        node.label = max(node.class_counts, key=node.class_counts.get)
        node.children = {}
        node.attribute = None
        return

    for val, child in node.children.items():
        idx = features[node.attribute] == val
        chi_squared_prune(child, features[idx], labels[idx], alpha)


def cross_validate(df, target, params, use_chi2=True):
    folds = stratified_k_fold(df, target, 10)
    scores = []

    for i in range(10):
        val = folds[i]
        train = pd.concat([folds[j] for j in range(10) if j != i])

        features_train, labels_train = train.drop(columns=[target]), train[target]
        features_val, labels_val = val.drop(columns=[target]), val[target]

        tree = ID3(params['max_depth'], params['min_samples'], params['min_gain'])
        tree.build(features_train, labels_train)
        
        if use_chi2:
            chi_squared_prune(tree.root, features_train, labels_train)

        preds = tree.predict(features_val)
        scores.append(accuracy(labels_val, preds))

    return scores


def main():
    np.random.seed(123)
    tokens = input().strip().split()
    mode = tokens[0]
    flags = set(tokens[1:])

    use_pre = mode in ["0", "2"]
    use_post = mode in ["1", "2"]
    max_depth = 6 if (not flags or "N" in flags) and use_pre else None
    min_samples = 5 if (not flags or "K" in flags) and use_pre else 1
    min_gain = 0.01 if (not flags or "G" in flags) and use_pre else 0

    use_chi2 = use_post and (not flags or "X" in flags)

    df = prep_missing(load_data())
    train_df, test_df = stratified_train_test_split(df, "Class")

    features_train, labels_train = train_df.drop(columns=["Class"]), train_df["Class"]
    features_test, labels_test = test_df.drop(columns=["Class"]), test_df["Class"]

    tree = ID3(max_depth, min_samples, min_gain)
    tree.build(features_train, labels_train)

    if use_chi2:
        chi_squared_prune(tree.root, features_train, labels_train)

    train_acc = accuracy(labels_train, tree.predict(features_train))
    test_acc = accuracy(labels_test, tree.predict(features_test))

    params = dict(max_depth=max_depth, min_samples=min_samples, min_gain=min_gain)
    cv = cross_validate(train_df, "Class", params, use_chi2)

    print("1. Train Set Accuracy:")
    print(f"    Accuracy: {train_acc:.2f}%\n")

    print("10-Fold Cross-Validation Results:\n")
    for i, a in enumerate(cv):
        print(f"    Accuracy Fold {i+1}: {a:.2f}%")

    print(f"\n    Average Accuracy: {np.mean(cv):.2f}%")
    print(f"    Standard Deviation: {np.std(cv):.2f}%\n")

    print("2. Test Set Accuracy:")
    print(f"    Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
