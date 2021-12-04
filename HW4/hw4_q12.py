import numpy as np
from liblinear.liblinearutil import *

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
    for i in range(2, Q + 1): 
        idx = set(range(len(data[0])))
        for j in range(len(data[0])):
            idx = idx - {j}
            temp = np.transpose(data[:, list(idx)])
            first = np.power(data[:, j], i - 1)
            for col in temp:
                data_tf = np.column_stack((data_tf, np.multiply(first, col)))

        exp = np.power(data, i)                  
        data_tf = np.column_stack((data_tf, exp))
    
    return data_tf

def ZeroOneError(X_pred, y):
    pred = [np.sign(e) for e in X_pred]
    y_pred = [np.sign(e) for e in pred]
    err = np.mean(y != y_pred)

    return err
  
X_train, y_train = readFile(train_filePath)
X_test, y_test = readFile(test_filePath)

X_train_tf = transform(X_train, 3)
X_test_tf = transform(X_test, 3)

print(X_train_tf.shape)
print(X_test_tf.shape)

minEout = np.inf
best_log10_lambda = 0
for log10_lambda in (-4, -2, 0, 2, 4):
    c = 0.5 * (1 / 10 ** log10_lambda)
    model = train(y_train, X_train_tf, '-s 0 -c ' + str(c) + ' -e 0.000001')
    Pred_test, _, _ = predict(y_test, X_test_tf, model)
    Eout = ZeroOneError(y_test, Pred_test)
     
    if Eout < minEout:
        minEout = Eout
        best_log10_lambda = log10_lambda
    else:
        best_log10_lambda = max(best_log10_lambda, log10_lambda)

print('The best log_10(λ*) = ', best_log10_lambda)
