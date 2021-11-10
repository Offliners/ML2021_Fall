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

  
Ein = []
expTimes = 100
for i in range(expTimes):
    coinData = flipCoin(200, i)
    X, y = coinData[:, :3], coinData[:, 3]
    dagger = np.linalg.pinv(X)
    wlin = np.matmul(dagger, y)
    Ein.append(np.square(np.matmul(X, wlin) - y).mean())

print(sum(Ein) / expTimes)
