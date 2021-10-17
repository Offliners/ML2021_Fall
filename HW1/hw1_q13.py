import numpy as np

filePath = 'hw1_train.dat'

def readFile(path):
    X = []
    y = []
    for line in open(path).readlines():
        data = line.strip().split('\t')
        y.append(float(data[-1]))
        X.append(list(map(float, data[:-1])))

    return np.array(X), np.array(y)

X, y = readFile(filePath)
X = np.insert(X, 0, np.ones(X.shape[0]), 1)

print(X.shape)
print(y.shape)

wpla_norm = 0
expTimes = 1000
for expTime in range(expTimes):
    np.random.seed(expTime)

    w = np.zeros(X.shape[1])
    noMistakeTime = 0
    while True:
        index = np.random.randint(0, X.shape[0])
        
        if np.sign(w.T.dot(X[index])) == 0:
            sign = -1
        else:
            sign = np.sign(w.T.dot(X[index]))

        if sign != y[index]:
            noMistakeTime = 0
            w += y[index] * X[index]
        else:
            noMistakeTime += 1

        if noMistakeTime == 5 * X.shape[0]:
            break

    wpla_norm += sum([e ** 2 for e in w])

print(wpla_norm / expTimes)
