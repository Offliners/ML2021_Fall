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

def transform(data) :
    index = np.random.randint(0, 10, 5)           
    data_tf = np.column_stack(((np.ones(len(data)), data[:, index])))
    
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

expTime = 200
err = 0
for i in range(expTime):
    np.random.seed(i)
    X_train_tf = transform(X_train)
    X_test_tf = transform(X_test)

    Pred_train, wlin = LinearRegression(X_train_tf, y_train)
    Ein = ZeroOneError(Pred_train, y_train)

    Pred_test = np.matmul(X_test_tf, wlin)
    Eout = ZeroOneError(Pred_test, y_test)

    err += abs(Ein - Eout)

print(err / expTime)
