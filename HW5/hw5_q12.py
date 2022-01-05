import numpy as np
from libsvm.svmutil import *

def relabel(dataset, target):
    new_dataset = []

    for item in dataset:
        new_dataset.append(1 if item == target else -1)
    
    return new_dataset
  
train_filePath = 'satimage.scale'

y_train, X_train = svm_read_problem(train_filePath)

maxEin = 0
ans = 0
for target in range(2, 7):
    new_y_train = relabel(y_train, target)

    model = svm_train(new_y_train, X_train, '-s 0 -c 10 -d 3 -t 1 -g 1 -r 1')
    p_label, p_acc, p_val = svm_predict(new_y_train, X_train, model)
    print(p_acc)
    Ein = 100 - p_acc[0]
    print(f'\"{target}\" versus \"not {target}\" : Ein = {Ein}')

    if Ein > maxEin:
        maxEin = Ein
        ans = target

print('=============================')
print(f'{ans} reaches the max Ein')
