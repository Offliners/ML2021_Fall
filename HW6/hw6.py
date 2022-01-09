import numpy as np

def readFile(path):
    X = []
    y = []
    for line in open(path).readlines():
        data = line.strip().split(' ')
        y.append(float(data[-1]))
        X.append(list(map(float, data[:-1])))

    return np.array(X), np.array(y)

def decisionStump(X, y, u):
    X_sorted = np.sort(X)
    thetas = [(X_sorted[i-1] + X_sorted[i]) / 2 if i != 0 else -np.inf for i in range(X.shape[0])]

    pocket_s = 1
    pocket_t = 1
    pocket_err = 1
    for t in thetas:
        s = 1
        y_pred = np.sign(X - t)
        err1 = np.sum(u * (y != y_pred)) / X.shape[0]
        err2 = np.sum(u * (y != -1 * y_pred)) / X.shape[0]

        if err1 > err2:
            err1 = err2
            s = -1
        
        if err1 < pocket_err:
            pocket_s = s
            pocket_t = t
            pocket_err = err1
    
    return pocket_s, pocket_t, pocket_err

def getBest_s_theta(X, y, u):
    pocket_s = 1
    pocket_t = 1
    pocket_err = 1
    pocket_feature = 0
    for i in range(X.shape[1]):
        s, t, err = decisionStump(X[:, i], y, u)

        if err < pocket_err:
            pocket_s = s
            pocket_t = t
            pocket_err = err
            pocket_feature = i
    
    return pocket_s, pocket_t, pocket_err, pocket_feature

def calGt(X, y, alpha, param, isUniform):
    vote = 0
    for i in range(len(alpha)):
        p_s, p_t, _, p_feature = param[i]
        y_predict = p_s * np.sign(X[:, p_feature] - p_t)

        if isUniform == True:
            a = 1
        else:
            a = alpha[i]

        vote += a * y_predict
    
    return np.sum(y != np.sign(vote)) / y.shape[0]

train_filePath = 'hw6_train.dat'
test_filePath = 'hw6_test.dat'

X_train, y_train = readFile(train_filePath)
X_test, y_test = readFile(test_filePath)

print(X_train.shape)
print(X_test.shape)

u = np.ones(X_train.shape[0]) / X_train.shape[0]
Ein_gts = []
Ein_Gts = []
Alphas = []
Params = []
expTimes = 500
for expTime in range(expTimes):
    best_s, best_t, err, select_feature = getBest_s_theta(X_train, y_train, u)
    Params.append([best_s, best_t, err, select_feature])
    y_pred = best_s * np.sign(X_train[:, select_feature] - best_t)
    et = err * X_train.shape[0] / np.sum(u)
    dt = np.sqrt((1 - et) / et)
    Alphas.append(np.log(dt))
    u = np.where(y_train != y_pred, u * dt, u / dt)
    Ein_gts.append(np.sum(y_train != y_pred) / X_train.shape[0])
    Ein_Gts.append(calGt(X_train, y_train, Alphas[:expTime], Params[:expTime], False))

# Q11
print(f'Q11 : {Ein_gts[0]}')

# Q12
print(f'Q12 : {max(Ein_gts)}')

# Q13
t = 0
for i in range(len(Ein_Gts)):
    if Ein_Gts[i] <= 0.05:
        t = i
        break

print(f'Q13 : {t + 1}')

# Q14
Eout_gt = []
Eout_Gt = []
Eout_Gt_uniform = []
for i in range(len(Alphas)):
    p_s, p_t, p_err, p_feature = Params[i]
    y_predict = p_s * np.sign(X_test[:, p_feature] - p_t)
    Eout = np.sum(y_test != y_predict) / y_test.shape[0]
    Eout_gt.append(Eout)
    Eout_Gt.append(calGt(X_test, y_test, Alphas[:i], Params[:i], False))
    Eout_Gt_uniform.append(calGt(X_test, y_test, Alphas[:i], Params[:i], True))

print(f'Q14 : {Eout_gt[0]}')

# Q15
print(f'Q15 : {Eout_Gt_uniform[-1]}')

# Q16
print(f'Q16 : {Eout_Gt[-1]}')
