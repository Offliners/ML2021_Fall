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

minEout = np.inf
ans = 0
for gamma in range(5):
    new_y_train = relabel(y_train, 1)
    new_y_test = relabel(y_test, 1)
    g = 10 ** (gamma - 1)
    model = svm_train(new_y_train, X_train, f'-s 0 -c 0.1 -t 2 -g {g}')
    p_label, p_acc, p_val = svm_predict(new_y_test, X_test, model)

    Eout = 100 - p_acc[0]
    print(f'Gamma = {g} : Eout = {Eout}')
    if Eout < minEout:
        minEout = Eout
        ans = g

print('=============================')
print(f'Gamma = {ans} reaches the min Eout')
