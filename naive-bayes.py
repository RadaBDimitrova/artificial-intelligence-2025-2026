import numpy as np
import pandas as pd

def load_data():
    columns = [
        'class', 'handicapped-infants', 'water-project-cost-sharing',
        'adoption-of-the-budget-resolution', 'physician-fee-freeze',
        'el-salvador-aid', 'religious-groups-in-schools',
        'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
        'mx-missile', 'immigration', 'synfuels-corporation-cutback',
        'education-spending', 'superfund-right-to-sue', 'crime',
        'duty-free-exports', 'export-administration-act-south-africa'
    ]
    data = pd.read_csv('house-votes-84.data', names=columns, na_values='?')
    return data

def preprocess_data(data, mode):
    df = data.copy()
    if mode == 0:
        df = df.fillna('a')
    else:
        # class-conditional imputation
        for col in df.columns[1:]:
            for class_label in df['class'].unique():
                mask = (df['class'] == class_label) & (df[col].isna())
                if mask.any():
                    mode_value = df[df['class'] == class_label][col].mode()
                    if len(mode_value) > 0:
                        df.loc[mask, col] = mode_value[0]
            if df[col].isna().any():
                global_mode = df[col].mode()
                if len(global_mode) > 0:
                    df[col] = df[col].fillna(global_mode[0])
    return df

def reset_indices(*dataframes):
    return tuple(df.reset_index(drop=True) for df in dataframes)

def stratified_train_test_split(features, labels, test_size=0.2):
    features, labels = reset_indices(features, labels)
    classes = labels.unique()
    train_indices = []
    test_indices = []
    
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        n_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return features.iloc[train_indices], features.iloc[test_indices], labels.iloc[train_indices], labels.iloc[test_indices]

class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probabilites = {}
        self.classes = None
        self.all_feature_values = {}
        self.class_counts = {}
        
    def train(self, features, labels):
        self.classes = labels.unique()
        n_samples = len(labels)
        for feature in features.columns:
            self.all_feature_values[feature] = features[feature].unique()
        
        # class priors P(C)
        for cls in self.classes:
            class_mask = labels == cls
            class_count = class_mask.sum()
            self.class_counts[cls] = class_count
            self.class_priors[cls] = np.log(class_count / n_samples)
        
        self.feature_probabilites = {}
        for cls in self.classes:
            features_cls = features[labels == cls]
            self.feature_probabilites[cls] = {}
            
            for feature in features.columns:
                feature_values = self.all_feature_values[feature]
                n_values = len(feature_values)
                n_cls_samples = len(features_cls)
                self.feature_probabilites[cls][feature] = {}
                for value in feature_values:
                    count = (features_cls[feature] == value).sum()
                    prob = np.log((count + self.alpha) / (n_cls_samples + self.alpha * n_values))
                    self.feature_probabilites[cls][feature][value] = prob
        return self
    
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
                        n_values = len(self.all_feature_values[feature])
                        n_cls_samples = self.class_counts[cls]
                        score += np.log(self.alpha / (n_cls_samples + self.alpha * n_values))
                
                class_scores[cls] = score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def score(self, features, labels):
        return (self.predict(features) == labels.values).mean()

def stratified_kfold_split(labels, n_splits=10):
    indices = np.arange(len(labels))
    classes = labels.unique()
    folds = [[] for _ in range(n_splits)]
    
    for cls in classes:
        cls_mask = (labels.values == cls)
        cls_indices = indices[cls_mask]
        np.random.shuffle(cls_indices)
        for i, idx in enumerate(cls_indices):
            folds[i % n_splits].append(idx)
    
    for fold_idx in range(n_splits):
        val_indices = folds[fold_idx]
        train_indices = [idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold]
        yield train_indices, val_indices

def evaluate_model(features_train, labels_train, features_test, labels_test):
    features_train, labels_train, features_test, labels_test = reset_indices(features_train, labels_train, features_test, labels_test)
    model = NaiveBayesClassifier(alpha=0.1)
    model.train(features_train, labels_train)
    train_accuracy = model.score(features_train, labels_train) * 100
    
    cv_scores = []
    for train_idx, val_idx in stratified_kfold_split(labels_train, n_splits=10):
        features_fold_train = features_train.iloc[train_idx]
        labels_fold_train = labels_train.iloc[train_idx]
        features_fold_val = features_train.iloc[val_idx]
        labels_fold_val = labels_train.iloc[val_idx]
        
        fold_model = NaiveBayesClassifier(alpha=0.1)
        fold_model.train(features_fold_train, labels_fold_train)
        fold_accuracy = fold_model.score(features_fold_val, labels_fold_val) * 100
        cv_scores.append(fold_accuracy)
    
    cv_mean = np.mean(cv_scores)
    cv_var = np.std(cv_scores)
    test_accuracy = model.score(features_test, labels_test) * 100
    
    return train_accuracy, cv_scores, cv_mean, cv_var, test_accuracy

def main():
    mode = int(input().strip())
    data = load_data()
    data = data.sample(frac=1, random_state=123).reset_index(drop=True)
    df_processed = preprocess_data(data, mode)
    
    features = df_processed.drop('class', axis=1)
    labels = df_processed['class']
    
    np.random.seed(123)
    features_train, features_test, labels_train, labels_test = stratified_train_test_split(features, labels, test_size=0.2)
    
    np.random.seed(2003)
    train_acc, cv_scores, cv_mean, cv_var, test_acc = evaluate_model(features_train, labels_train, features_test, labels_test)
    
    print("1. Train Set Accuracy:")
    print(f"    Accuracy: {train_acc:.2f}%\n")
    
    print("10-Fold Cross-Validation Results:\n")
    for i, score in enumerate(cv_scores, 1):
        print(f"    Accuracy Fold {i}: {score:.2f}%")
    
    print(f"\n    Average Accuracy: {cv_mean:.2f}%")
    print(f"    Standard Deviation: {cv_var:.2f}%")
    print("\n2. Test Set Accuracy:")
    print(f"    Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
