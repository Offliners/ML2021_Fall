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

X_train_tf_split, y_train_split = X_train_tf[:120], y_train[:120]
X_val, y_val = X_train_tf[120:], y_train[120:]

print(X_train_tf_split.shape)
print(X_val.shape)

minEval = np.inf
best_log10_lambda = 0
best_model = None
for log10_lambda in (-4, -2, 0, 2, 4):
    c = 1 / 2 / (10 ** log10_lambda)
    model = train(y_train_split, X_train_tf_split, '-s 0 -c ' + str(c) + ' -e 0.000001')
    Pred_val, _, _ = predict(y_val, X_val, model, ' -q')
    Eval = ZeroOneError(Pred_val, y_val)

    if Eval == minEval and log10_lambda > best_log10_lambda:
        best_log10_lambda = log10_lambda
        best_model = model
    elif Eval < minEval:
        minEval = Eval
        best_log10_lambda = log10_lambda
        best_model = model

Pred_test, _, _ = predict(y_test, X_test_tf, best_model)
Eout = ZeroOneError(Pred_test, y_test)

print('Eout = ', Eout)
