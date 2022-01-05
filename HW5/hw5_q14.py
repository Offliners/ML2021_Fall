import numpy as np
from libsvm.svmutil import *

def relabel(dataset, target):
    new_dataset = []

    for item in dataset:
        new_dataset.append(1 if item == target else -1)
    
    return new_dataset
  
train_filePath = 'satimage.scale'
test_filePath = 'satimage.scale.t'

y_train, X_train = svm_read_problem(train_filePath)
y_test, X_test = svm_read_problem(test_filePath)
X_upper = max(X_train[0].keys())
X_lower = min(X_train[0].keys())

minEout = np.inf
ans = 0
for exp in range(5):
    new_y_train = relabel(y_train, 1)
    new_y_test = relabel(y_test, 1)
    c = 10 ** (exp - 2)
    model = svm_train(new_y_train, X_train, f'-s 0 -c {c} -t 2 -g 10')
    p_label, p_acc, p_val = svm_predict(new_y_test, X_test, model)

    Eout = 100 - p_acc[0]
    print(f'C = {c} : Eout = {Eout}')
    if Eout < minEout:
        minEout = Eout
        ans = c

print('=============================')
print(f'C = {ans} reaches the min Eout')
