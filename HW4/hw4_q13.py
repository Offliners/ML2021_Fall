import numpy as np
from liblinear.liblinearutil import *
from itertools import combinations_with_replacement

train_filePath = 'hw4_train.dat'
test_filePath = 'hw4_test.dat'

def readFile(path):
    X = []
    y = []
    for line in open(path).readlines():
        data = line.strip().split(' ')
        y.append(float(data[-1]))
        X.append(list(map(float, data[:-1])))

    return np.array(X), np.array(y)

def transform(data, Q) : 
    data_tf = np.column_stack((np.ones(len(data)), data))
    c = range(len(data[0]))
    for i in range(2, Q + 1):
        orderList = [e for e in combinations_with_replacement(c, i)]
        for j in orderList:
            temp = np.ones(len(data))
            for k in range(len(j)):
                temp = np.multiply(temp, data[:, j[k]])
            
            data_tf = np.column_stack((data_tf, temp))
        
    return data_tf

def ZeroOneError(X_pred, y):
    err = np.mean(y != X_pred)

    return err

X_train, y_train = readFile(train_filePath)
X_test, y_test = readFile(test_filePath)

X_train_tf = transform(X_train, 3)
X_test_tf = transform(X_test, 3)

print(X_train_tf.shape)
print(X_test_tf.shape)

minEin = np.inf
best_log10_lambda = 0
for log10_lambda in (-4, -2, 0, 2, 4):
    c = 1 / 2 / (10 ** log10_lambda)
    model = train(y_train, X_train_tf, '-s 0 -c ' + str(c) + ' -e 0.000001')
    Pred_train, _, _ = predict(y_train, X_train_tf, model)
    Ein = ZeroOneError(Pred_train, y_train)

    if Ein == minEin:
        best_log10_lambda = max(best_log10_lambda, log10_lambda)
    elif Ein < minEin:
        minEin = Ein
        best_log10_lambda = log10_lambda

print('The best log_10(λ*) = ', best_log10_lambda)
