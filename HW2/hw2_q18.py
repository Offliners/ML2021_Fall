import numpy as np

def flipCoin(dataNum, seed):
    np.random.seed(seed)

    coin = np.random.randint(0, 2, dataNum)
    coin = coin * 2 - 1
    data = np.empty((dataNum, 4))
    for i in range(dataNum):
        data[i][0] = 1
        if coin[i] == 1:
            data[i][1:3] = np.random.multivariate_normal(mean=[2, 3], cov=[[0.6, 0], [0, 0.6]], size=1)
            data[i][3] = 1
        else:
            data[i][1:3] = np.random.multivariate_normal(mean=[0, 4], cov=[[0.4, 0], [0, 0.4]], size=1)
            data[i][3] = -1

    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SGD(X, y, learning_rate, weight, epoch):
    for i in range(epoch):
        grad = 0
        for i in range(X.shape[0]):
            grad += sigmoid(-1 * y[i] * np.dot(weight.T, X[i])) * (y[i] * X[i]) 
        weight += learning_rate * grad / X.shape[0]
    
    return weight

def genearteOutlier(dataNum, seed):
    data = np.empty((dataNum, 4))
    
    for i in range(dataNum):
        data[i][0] = 1
        data[i][1:3] = np.random.multivariate_normal(mean=[6, 0], cov=[[0.3, 0], [0, 0.1]], size=1)
        data[i][3] = 1
    
    return data
  
Eout_linear_regression = []
Eout_logistic_regression = []
expTimes = 100
for i in range(expTimes):
    coinData_train = flipCoin(200, i)
    coinData_outlier = genearteOutlier(20, i)
    coinData_train = np.concatenate((coinData_train, coinData_outlier), axis=0)
    coinData_test = flipCoin(5000, i)
    X_train, y_train = coinData_train[:, :3], coinData_train[:, 3]
    X_test, y_test = coinData_test[:, :3], coinData_test[:, 3]

    dagger = np.linalg.pinv(X_train)
    wlin = np.matmul(dagger, y_train)
    Pred_train = np.matmul(X_train, wlin)
    Pred_test = np.matmul(X_test, wlin)
    Pred_test = [np.sign(e) for e in Pred_test]
    Eout = np.mean(Pred_test != y_test)
    Eout_linear_regression.append(Eout)

    w = np.zeros(3)
    w = SGD(X_train, y_train, 0.1, w, 500)
    Pred_test = np.sign(sigmoid(np.dot(X_test, w)) - 0.5)
    Eout = np.mean(Pred_test != y_test)
    Eout_logistic_regression.append(Eout)

print(sum(Eout_linear_regression) / expTimes)
print(sum(Eout_logistic_regression) / expTimes)
