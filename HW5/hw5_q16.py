import numpy as np
from libsvm.svmutil import *

def relabel(dataset, target):
    new_dataset = []

    for item in dataset:
        new_dataset.append(1 if item == target else -1)
    
    return new_dataset

def shuffle(x, y, N):
    p = np.random.permutation(len(y))
    X_shuffle = np.array(x)[p]
    y_shuffle = np.array(y)[p]
    return X_shuffle[N:], y_shuffle[N:], X_shuffle[:N], y_shuffle[:N]

train_filePath = 'satimage.scale'

y_train, X_train = svm_read_problem(train_filePath)

minEout = np.inf
ans = 0
expTimes = 1000
best_gamma_list = {0.1:0, 1:0, 10:0, 100:0, 1000:0}
for expTime in range(expTimes):
    np.random.seed(expTime)
    new_y_train = relabel(y_train, 1)
    X_train_shuffle, y_train_shuffle, X_val, y_val = shuffle(X_train, new_y_train, 200)

    minEval = np.inf
    best_g = 0
    for gamma in range(5):
        g = 10 ** (gamma - 1)
        model = svm_train(y_train_shuffle, X_train_shuffle, f'-s 0 -c 0.1 -t 2 -g {g} -q')
        p_label, p_acc, p_val = svm_predict(y_val, X_val, model, '-q')

        Eval = 100 - p_acc[0]
        if Eval < minEval:
            minEval = Eval
            best_g = g
    
    best_gamma_list[best_g] += 1

print(best_gamma_list)
maxTimes = 0
best_gamma = 0
for g, times in best_gamma_list.items():
    if times > maxTimes:
        maxTimes = times
        best_gamma = g

print('=============================')
print(f'Gamma selected the most number of times is {best_gamma}')
