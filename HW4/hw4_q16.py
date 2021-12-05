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
  
def CrossValidation(X, y, V):
    return np.vstack((X[:V * 40],X[(V + 1) * 40:])),\
           np.hstack((y[:V * 40],y[(V + 1) * 40:])),\
           X[V * 40:(V + 1) * 40],\
           y[V * 40:(V + 1) * 40]

X_train, y_train = readFile(train_filePath)
X_test, y_test = readFile(test_filePath)

X_train_tf = transform(X_train, 3)
X_test_tf = transform(X_test, 3)

X_train_tf_split, y_train_split = X_train_tf[:120], y_train[:120]
X_val, y_val = X_train_tf[120:], y_train[120:]

print(X_train_tf_split.shape)
print(X_val.shape)

minEcv = np.inf
best_log10_lambda = 0
best_Ecv = 0
V = 5
for log10_lambda in (-4, -2, 0, 2, 4):
    Ecv = 0
    C = 1 / 2 / (10 ** log10_lambda)
    for v in range(V):
        X_train_split, y_train_split, X_val, y_val = CrossValidation(X_train_tf, y_train, v)
        model = train(y_train_split, X_train_split, '-s 0 -c ' + str(C) + ' -e 0.000001 -q')
        Pred_val, _, _ = predict(y_val, X_val, model, ' -q')
        Eval = ZeroOneError(Pred_val, y_val)
        Ecv += Eval

    Ecv = Ecv / V
    if Ecv == minEcv and log10_lambda > best_log10_lambda:
        best_log10_lambda = log10_lambda
        best_Ecv = Ecv
    elif Ecv < minEcv:
        minEcv = Ecv
        best_log10_lambda = log10_lambda
        best_Ecv = Ecv

print('The best Ecv = ', best_Ecv)
