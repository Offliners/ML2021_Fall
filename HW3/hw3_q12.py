import numpy as np

train_filePath = 'hw3_train.dat'
test_filePath = 'hw3_test.dat'

def readFile(path):
    X = []
    y = []
    for line in open(path).readlines():
        data = line.strip().split('\t')
        y.append(float(data[-1]))
        X.append(list(map(float, data[:-1])))

    return np.array(X), np.array(y)

def transform(data, Q) : 
    data_tf = np.column_stack((np.ones(len(data)), data))
    for i in range(2, Q + 1): 
        exp = np.power(data, i)                  
        data_tf = np.column_stack((data_tf, exp)) 
    
    return data_tf

def LinearRegression(X, y):
    dagger = np.linalg.pinv(X)
    wlin = np.matmul(dagger, y)
    pred = np.matmul(X, wlin)
    
    return pred, wlin

def ZeroOneError(X_pred, y):
    pred = [np.sign(e) for e in X_pred]
    y_pred = [np.sign(e) for e in pred]
    err = np.mean(y != y_pred)

    return err



X_train, y_train = readFile(train_filePath)
X_test, y_test = readFile(test_filePath)

X_train_tf = transform(X_train, 2)
X_test_tf = transform(X_test, 2)

print(X_train_tf.shape)
print(X_test_tf.shape)

Pred_train, wlin = LinearRegression(X_train_tf, y_train)
Ein = ZeroOneError(Pred_train, y_train)

Pred_test = np.matmul(X_test_tf, wlin)
Eout = ZeroOneError(Pred_test, y_test)

print(abs(Ein - Eout))
