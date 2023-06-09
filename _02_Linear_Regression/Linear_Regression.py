# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x_train, y_train = read_data()
    x_train = np.column_stack((np.ones(len(x_train)), x_train))[:, 1:]

    alpha = -500
    # 计算正则化系数矩阵
    l = alpha * np.eye(x_train.shape[1])

    weight = np.linalg.solve(np.dot(x_train.T, x_train) + l, np.dot(x_train.T, y_train))
    return np.sum(data * weight)


def lasso(data):
    x,y=read_data()
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