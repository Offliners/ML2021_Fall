import numpy as np
from libsvm.svmutil import *

def relabel(dataset, target):
    new_dataset = []

    for item in dataset:
        new_dataset.append(1 if item == target else -1)
    
    return new_dataset

def sparse_matrix_dense(dataset, lower, upper):
    dense = [dataset.get(item) for item in range(lower, upper + 1)]
    return np.array([0 if item == None else item for item in dense])
  
train_filePath = 'satimage.scale'

y_train, X_train = svm_read_problem(train_filePath)
upper = max(X_train[0].keys())
lower = min(X_train[0].keys())
new_y_train = relabel(y_train, 5)

model = svm_train(new_y_train, X_train, '-c 10 -t 0')
w = np.zeros(upper)
for i in range(model.l):
    w += model.sv_coef[0][i] * sparse_matrix_dense(X_train[model.sv_indices[i] - 1], lower, upper)

print(np.linalg.norm(w))