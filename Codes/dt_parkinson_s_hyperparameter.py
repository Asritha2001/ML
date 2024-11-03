import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left  # Left child
        self.right = right  # Right child
        self.value = value  # Value to return if node is a leaf


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def information_gain(X, y, feature_index, threshold):
    parent_entropy = entropy(y)
    left_indices = X.iloc[:, feature_index] < threshold
    right_indices = ~left_indices
    n_left, n_right = np.sum(left_indices), np.sum(right_indices)
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    child_entropy = (n_left * left_entropy + n_right * right_entropy) / len(y)
    return parent_entropy - child_entropy


def split_dataset(X, y, feature_index, threshold):
    left_indices = X.iloc[:, feature_index] < threshold
    right_indices = ~left_indices
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]


def find_best_split(X, y):
    best_info_gain = -1
    best_feature_index = None
    best_threshold = None
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X.iloc[:, feature_index])
        for threshold in thresholds:
            info_gain = information_gain(X, y, feature_index, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold


def build_tree(X, y, depth, max_depth):
    if depth == max_depth or len(np.unique(y)) == 1:
        return Node(value=np.argmax(np.bincount(y)))
    best_feature_index, best_threshold = find_best_split(X, y)
    if best_feature_index is None:
        return Node(value=np.argmax(np.bincount(y)))
    left_X, left_y, right_X, right_y = split_dataset(X, y, best_feature_index, best_threshold)
    left_child = build_tree(left_X, left_y, depth + 1, max_depth)
    right_child = build_tree(right_X, right_y, depth + 1, max_depth)
    return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_child, right=right_child)


def predict(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature_index] < node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)


def stratified_cross_validation(X, y, num_folds):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    folds = np.array_split(indices, num_folds)
    for i in range(num_folds):
        train_indices = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        test_indices = folds[i]
        yield train_indices, test_indices


# Load the Parkinson's Dataset
data = pd.read_csv('parkinsons.csv')
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

depth_values = [3, 5, 7, 9, 12]
num_folds = 10

accuracies = []
f1_scores = []

for depth in depth_values:
    fold_accuracies = []
    fold_f1_scores = []
    for train_indices, test_indices in stratified_cross_validation(X, y, num_folds):
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        tree = build_tree(X_train, y_train, depth=0, max_depth=depth)
        y_pred = np.array([predict(tree, sample) for sample in X_test.values])
        fold_accuracies.append(accuracy_score(y_test.values, y_pred))
        fold_f1_scores.append(f1_score(y_test.values, y_pred))
    accuracies.append(np.mean(fold_accuracies))
    f1_scores.append(np.mean(fold_f1_scores))

# Plot accuracy and F1 score as a function of depth
plt.plot(depth_values, accuracies, label='Accuracy')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Depth')
plt.legend()
plt.show()

plt.plot(depth_values, f1_scores, label='F1 Score')
plt.xlabel('Depth')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Depth')
plt.legend()
plt.show()

# Print summary statistics
print("Depth\tAccuracy\tF1 Score")
for i in range(len(depth_values)):
    print(f"{depth_values[i]}\t{accuracies[i]}\t{f1_scores[i]}")
