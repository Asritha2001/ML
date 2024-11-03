import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


# Decision Tree Definition
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value (class) if the node is a leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < 2:
            return Node(value=np.argmax(np.bincount(y)))

        # Finding the best split
        best_gain = -np.inf
        best_feature_index = None
        best_threshold = None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        # Creating sub-nodes
        left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold,
                    left=left_child, right=right_child)

    def _information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        entropy_parent = self._entropy(y)
        entropy_children = p * self._entropy(y_left) + (1 - p) * self._entropy(y_right)
        return entropy_parent - entropy_children

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)


# One-hot encoding for categorical attributes
def one_hot_encode(df):
    return pd.get_dummies(df)


# Implementing stratified cross-validation
def stratified_cross_validation(X, y, clf, k=10):
    n_samples = len(y)
    fold_size = n_samples // k
    accuracy_scores = []
    f1_scores = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n_samples

        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        X_test = X[start:end]
        y_test = y[start:end]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return accuracy_scores, f1_scores


# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

depth_values = [3, 5, 7, 9, 12]
# depth_values = [10]

# Evaluate performance for each depth value
accuracy_results = []
f1_results = []

for depth in depth_values:
    clf = DecisionTreeClassifier(max_depth=depth)
    accuracy_scores, f1_scores = stratified_cross_validation(X, y, clf)
    accuracy_results.append(np.mean(accuracy_scores))
    f1_results.append(np.mean(f1_scores))

# Plot accuracy and F1 score as a function of depth
plt.figure(figsize=(10, 5))
plt.plot(depth_values, accuracy_results, label='Accuracy')
plt.title('Accuracy vs Depth')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(depth_values, f1_results, label='F1 Score')
plt.title('F1 Score vs Depth')
plt.xlabel('Depth')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Summary statistics
print("Summary Statistics:")
for depth, accuracy, f1 in zip(depth_values, accuracy_results, f1_results):
    print(f"Depth: {depth}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")


