import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Decision Tree Definition
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (depth >= self.max_depth or n_classes == 1 or n_samples < 2):
            return Node(value=np.argmax(np.bincount(y)))

        best_gain = -np.inf
        best_feature_index = None
        best_threshold = None
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # Skip if splitting results in an empty child

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        if best_feature_index is None:
            return Node(value=np.argmax(np.bincount(y)))

        left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]
        left_child = None
        right_child = None

        if len(left_indices) > 0:
            left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        if len(right_indices) > 0:
            right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # If both splits are empty, prevent further splitting by directly returning a leaf node
        if left_child is None and right_child is None:
            return Node(value=np.argmax(np.bincount(y)))

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_child, right=right_child)

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


# Stratified cross-validation over multiple iterations
def repeated_evaluation(X, y, depth, iterations=100):
    accuracy_scores = []
    f1_scores = []
    skf = StratifiedKFold(n_splits=10)

    for _ in range(iterations):
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    return accuracy_scores, f1_scores


# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Perform the evaluations
accuracy_scores, f1_scores = repeated_evaluation(X, y, depth=10)

# Plotting histograms
plt.figure(figsize=(10, 5))
plt.hist(accuracy_scores, bins=10, color='lightblue', alpha=0.7)
plt.title('Histogram of Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(f1_scores, bins=10, color='lightblue', alpha=0.7)
plt.title('Histogram of F1 Scores')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.show()

# Summary statistics
print("Average Accuracy:", np.mean(accuracy_scores))
print("Standard Deviation of Accuracy:", np.std(accuracy_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Standard Deviation of F1 Score:", np.std(f1_scores))
