# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x_train, y_train = read_data(path='H:/linear-regression-3q3z4/data/exp02/')
    x_train = np.column_stack((np.ones(len(x_train)), x_train))[:, 1:]

    alpha = -500
    # 计算正则化系数矩阵
    l = alpha * np.eye(x_train.shape[1])

    weight = np.linalg.solve(np.dot(x_train.T, x_train) + l, np.dot(x_train.T, y_train))
    return np.sum(data * weight)


def lasso(data):
    x,y=read_data(path='H:/linear-regression-3q3z4/data/exp02/')
    m, n = x.shape
    max_iter=1000000
    lamda=1e-32
    alpha=1e-7
    theta = np.zeros(n)
    for i in range(404):
        x[i][3]=x[i][3]*x[i][3]
    for i in range(max_iter):
        h_theta = np.dot(x, theta)
        # 计算偏导数
        diff = np.zeros(n)
        for j in range(n):
            if theta[j] != 0:
                diff[j] = (1 / m) * np.dot(x[:, j].T, (h_theta - y)) + abs(lamda / (2 * m))
            else:
                diff[j] = np.random.uniform(-lamda / (2 * m), lamda / (2 * m))
        # 更新参数
        theta -= alpha * diff
    data[3]=data[3]**2
    return np.sum(data*theta)


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
features = np.array([
    [2.0133330e+03, 1.6400000e+01, 2.8932480e+02, 5.0000000e+00, 2.4982030e+01, 1.2154348e+02],
    [2.0126670e+03, 2.3000000e+01, 1.3099450e+02, 6.0000000e+00, 2.4956630e+01, 1.2153765e+02],
    [2.0131670e+03, 1.9000000e+00, 3.7213860e+02, 7.0000000e+00, 2.4972930e+01, 1.2154026e+02],
    [2.0130000e+03, 5.2000000e+00, 2.4089930e+03, 0.0000000e+00, 2.4955050e+01, 1.2155964e+02],
    [2.0134170e+03, 1.8500000e+01, 2.1757440e+03, 3.0000000e+00, 2.4963300e+01, 1.2151243e+02],
    [2.0130000e+03, 1.3700000e+01, 4.0820150e+03, 0.0000000e+00, 2.4941550e+01, 1.2150381e+02],
    [2.0126670e+03, 5.6000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02],
    [2.0132500e+03, 1.8800000e+01, 3.9096960e+02, 7.0000000e+00, 2.4979230e+01, 1.2153986e+02],
    [2.0130000e+03, 8.1000000e+00, 1.0481010e+02, 5.0000000e+00, 2.4966740e+01, 1.2154067e+02],
    [2.0135000e+03, 6.5000000e+00, 9.0456060e+01, 9.0000000e+00, 2.4974330e+01, 1.2154310e+02]
    ])

labels = np.array([41.2, 37.2, 40.5, 22.3, 28.1, 15.4, 50. , 40.6, 52.5, 63.9])
for i in range(10):
    print(abs(ridge(features[i]) - labels[i]), end=" ")
print()
for i in range(10):
    print(abs(lasso(features[i])-labels[i]),end=" ")