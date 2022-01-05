import numpy as np
from libsvm.svmutil import *

def relabel(dataset, target):
    new_dataset = []

    for item in dataset:
        new_dataset.append(1 if item == target else -1)
    
    return new_dataset
  
train_filePath = 'satimage.scale'

y_train, X_train = svm_read_problem(train_filePath)
X_upper = max(X_train[0].keys())
X_lower = min(X_train[0].keys())
new_y_train = relabel(y_train, 5)

model = svm_train(new_y_train, X_train, '-s 0 -c 10 -t 0')
w = np.zeros(X_upper)
for i in range(len(model.get_sv_coef())):
    support_vector = [model.get_SV()[i].get(item) for item in range(X_lower, X_upper + 1)]
    support_vector = np.array([0 if item == None else item for item in support_vector])
    w += model.get_sv_coef()[i][0] * support_vector

print(np.linalg.norm(w))
