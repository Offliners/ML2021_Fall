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

diff = []
expTimes = 100
for i in range(expTimes):
    coinData_train = flipCoin(200, i)
    coinData_test = flipCoin(5000, i)
    X_train, y_train = coinData_train[:, :3], coinData_train[:, 3]
    X_test, y_test = coinData_test[:, :3], coinData_test[:, 3]

    dagger = np.linalg.pinv(X_train)
    wlin = np.matmul(dagger, y_train)
    Pred_train = np.matmul(X_train, wlin)
    Pred_train = [np.sign(e) for e in Pred_train]
    Ein = np.mean(Pred_train != y_train)

    Pred_test = np.matmul(X_test, wlin)
    Pred_test = [np.sign(e) for e in Pred_test]
    Eout = np.mean(Pred_test != y_test)

    diff.append(abs(Ein - Eout))
    
print(sum(diff) / expTimes)
