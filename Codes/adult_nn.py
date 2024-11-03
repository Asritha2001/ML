# -*- coding: utf-8 -*-
"""adult_NN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UPtWcwRzTieuAjwQA5VFGkk6cKD5CuOL
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.seterr(divide = 'ignore')
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, f1_score

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
# Load the dataset
data = pd.read_csv("adult.data", names=column_names, sep=",\s*", engine='python')

sampled_data = data.sample(n=2000, random_state=42) # Randomly select 1000 instances
# Replace '?' with NaN to mark missing values
sampled_data.replace('?', pd.NA, inplace=True)
# You can choose to drop these rows or fill them
sampled_data.dropna(inplace=True)
X = sampled_data.iloc[:,0:-1]
y = sampled_data.iloc[:,-1].values.reshape(-1,1)
X = pd.get_dummies(X)


class NeuralNet:
  def __init__(self, layers, regularization_param):
    self.layers = layers
    self.regularization_param = regularization_param
    thetha = []
    for i in range(len(self.layers) - 1):
      w = np.random.uniform(-1, 1, (self.layers[i] + 1, self.layers[i + 1]))
      thetha.append(w)
    self.thetha = thetha

  def g_function(self,g):
    return np.round(1/(1 + np.exp(-g)),5)

  def forward_prop(self,X):
    activations = [X]
    for i, W in enumerate(self.thetha):
        # Adding bias
        X = np.insert(X, 0, 1, axis=1)
        Z = np.dot(X, W)
        A = self.g_function(Z)
        activations.append(A)
        X = A
    return activations

  def cost_instance(self,y_pred,y):
    cost = []
    for i in range(len(y)):
      cost_sample = 0
      for j in range(len(y[i])):
        cost_sample += -y[i][j]*np.log(y_pred[i][j]) - (1-y[i][j])*np.log(1-y_pred[i][j])
      cost.append(cost_sample/len(y[i]))
    return np.sum(cost)/len(y)

  def back_prop(self, X, y, alpha):
    act_x = self.forward_prop(X)
    y_pred = act_x[-1]
    delta = y_pred - y
    for i in range(len(self.thetha)-1,-1,-1):
      act = act_x[i]
      # bias adding
      act = np.insert(act,0,1,axis = 1)
      gradient = act.T.dot(delta) + self.regularization_param
      if i!=0:
        delta = (delta.dot(self.thetha[i][1:].T)) * act[:, 1:] * (1 - act[:, 1:])
      self.thetha[i] -= alpha * gradient

def stratificationKfolds(y,kfolds):
  total_instances = len(y)
  # dividing indices into seperate unique lists
  c1 = []
  c2 = []
  class_splits = []
  for i in range(total_instances):
    if y[i] == 0:
      c1.append(i)
    else:
      c2.append(i)
  classes = [c1,c2]
  # randomizing the class indices and splitting it
  for cl in classes:
    np.random.shuffle(cl)
    class_splits.append(np.array_split(cl, kfolds))
  # forming data into folds
  splits = []
  for fold in range(kfolds):
    test_indices = np.concatenate([class_splits[class_idx][fold] for class_idx in range(2)])
    train_indices = np.concatenate([np.concatenate(class_splits[class_idx][:fold] + class_splits[class_idx][fold + 1:]) for class_idx in range(2)])
    splits.append((train_indices, test_indices))
  return splits

def calculate_accuracy_f1score(y_true, y_pred):
  y_true_class = np.argmax(y_true, axis=1)
  y_pred_class = np.argmax(y_pred, axis=1)
  num_classes = len(y_true[0])
  confusion_matrix = np.zeros((num_classes, num_classes))
  x = 1e-6
  for i in range(len(y_true_class)):
    confusion_matrix[y_true_class[i]-1][y_pred_class[i]-1] += 1
  # print(confusion_matrix)
  confusion_matrix = np.array(confusion_matrix)
  # Calculate precision, recall, and F1 score for each class
  f1_scores = []
  accuracy = 0
  for i in range(num_classes):
    true_positives = confusion_matrix[i, i]
    accuracy += true_positives
    false_positives = np.sum(confusion_matrix[:, i]) - true_positives
    false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
    precision = true_positives / (true_positives + false_positives + x)
    recall = true_positives / (true_positives + false_negatives + x)
    f1_score = 2 * (precision * recall) / (precision + recall + x)
    f1_scores.append(f1_score)
  # Calculate the average F1 score for all classes
  avg_f1_score = np.mean(f1_scores)
  total_accuracy = accuracy/(len(y_true))
  return total_accuracy,avg_f1_score

def evaluate_model_on_training(X,onehot_labels,layers):
  X_train1, X_test1, y_train1, y_test1 = train_test_split(X, onehot_labels, test_size=0.2, random_state=42)
  performance_loss = []
  performance_accuracy = []
  samples = []
  for i in range(1, len(X_train1)+1):
    X_train_sample = X_train1[:i]
    y_train_sample = y_train1[:i]
    J,score,f1_score = evaluate_model(X_train_sample, y_train_sample, X_test1, y_test1,layers)
    # Print performance every 10 samples for progress tracking
    if i % 5 == 0:
        samples.append(i)
        performance_loss.append(J)
        performance_accuracy.append(score)
        print(f"Loss after {i} samples: {J}")
        # print(f"Accuracy after {i} samples: {score}")
  plt.plot(samples, performance_loss)
  plt.xticks(samples)
  plt.xlabel('Number of training samples')
  plt.ylabel('Performance (J) on the test set')
  plt.title('Learning Curve')
  plt.show()

  plt.plot(samples, performance_accuracy)
  plt.xticks(samples)
  plt.xlabel('Number of training samples')
  plt.ylabel('Accuracy on the test set')
  plt.title('Learning Curve')
  plt.show()

def normalize(df):
  normalized = (df.astype(np.float32) - df.min().astype(np.float32)) / (df.max().astype(np.float32) - df.min().astype(np.float32))
  return normalized

def stratified_evaluation(data,labels,y,kfolds,layers):
  scores = []
  f1_scores = []
  splits = stratificationKfolds(y,kfolds)
  for train_index, test_index in splits:
    X_train, X_test = data[train_index.astype(int)], data[test_index.astype(int)]
    y_train, y_test = labels[train_index.astype(int)], labels[test_index.astype(int)]
    J,score,f1_score = evaluate_model(X_train, y_train, X_test, y_test,layers)
    scores.append(score)
    f1_scores.append(f1_score)
  return np.mean(scores),np.mean(f1_scores)

def evaluate_model(X_train, y_train, X_test, y_test,layers) :
  epochs=1000
  learning_rate=0.01
  regularization_param=0.001
  NN = NeuralNet(layers, regularization_param)
  for i in range(epochs):
    NN.back_prop(X_train, y_train, learning_rate)
  a = NN.forward_prop(X_test)
  y_pred = a[-1]
  J = NN.cost_instance(y_pred,y_test)
  accuracy, f1_score = calculate_accuracy_f1score(y_test, y_pred)
  return J,accuracy,f1_score

def train(x,y):
  X = np.array(normalize(x))
  unique_values = np.unique(np.array(y))
  onehot_labels = []
  for i in range(len(y)):
    yi = [0]*len(unique_values)
    for j in range(len(unique_values)):
        if y[i] == unique_values[j]:
            yi[j] = 1
    onehot_labels.append(yi)
  onehot_labels = np.array(onehot_labels)
  input = X.shape[1]
  output = onehot_labels.shape[1]
  layers=[input,4,7, output]
  kfolds = 10
  accuracy_score, f1_score = stratified_evaluation(X, onehot_labels,y, kfolds, layers)
  print("Avg_Accuracy: ",accuracy_score)
  print("Avg_F1_score: ",f1_score)
  evaluate_model_on_training(X,onehot_labels,layers)

train(X, y)