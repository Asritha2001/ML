import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

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

data = pd.read_csv('parkinsons.csv')
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

num_iterations = 100
depth = 7
all_accuracies = []
all_f1_scores = []


for _ in range(num_iterations):
    shuffled_indices = np.random.permutation(len(y))
    split_index = int(len(y) * 0.8)
    train_indices = shuffled_indices[:split_index]
    test_indices = shuffled_indices[split_index:]
    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    tree = build_tree(X_train, y_train, 0, depth)
    y_pred = np.array([predict(tree, row) for index, row in X_test.iterrows()])
    all_accuracies.append(accuracy_score(y_test, y_pred))
    all_f1_scores.append(f1_score(y_test, y_pred))

# Print average and standard deviation
print("Average Accuracy: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(all_accuracies), np.std(all_accuracies)))
print("Average F1 Score: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(all_f1_scores), np.std(all_f1_scores)))

# Plot histograms
plt.hist(all_accuracies, bins=10, alpha=0.7, color='lightblue')
plt.title('Histogram of Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

plt.hist(all_f1_scores, bins=10, alpha=0.7, color='lightblue')
plt.title('Histogram of F1 Scores')
plt.xlabel('F1 Score')
plt.ylabel('Frequency')
plt.show()


