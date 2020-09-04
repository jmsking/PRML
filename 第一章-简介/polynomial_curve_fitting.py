import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class Dataset:
    """
    数据生成器
    """
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def gen_data(self):
        """
        产生数据
        """
        X = np.linspace(0, 1, self.n_samples, endpoint=True)
        Y = np.sin(2*np.pi*X)
        return X, Y
    
    def add_noise(self, Y):
        """
        添加高斯噪声
        """
        for i in range(Y.size):
            noise = random.gauss(mu=0.0, sigma=0.1)
            Y[i] += noise
        return Y

class PolynomialCurveFitting:

    def __init__(self):
        self.threshold = 1e-8
        self.lr = 0.0002
        self.epochs = 5000000

    def fit(self, X, Y, M):
        if M == 0:
            return np.random.randn(1)
        ori_X = X.copy()
        for i in range(2, M+1):
            X = np.concatenate((X, ori_X**i), axis=1)
        X = np.concatenate((np.ones((Y.size,1)), X), axis=1)
        print(X.shape)
        # 初始化权重
        w = np.random.randn(M+1)
        t = 0
        x, y = [], []
        while t < self.epochs:
            pred = np.matmul(X, w)
            diff = pred - Y
            error = 0.5 * np.matmul(diff.T, diff) / Y.size
            delta = np.matmul(diff.T, X)
            # 更新权重
            w = w - self.lr * delta
            t += 1
            x.append(t)
            y.append(error)
            #print(f'{t}/{self.epochs} - MSE: {error}')
            if error < self.threshold:
                break
        print('Training success!!!')
        print(error)
        """plt.plot(x, y, 'r')
        plt.show()"""
        return w

    def gen_image(self, M, w):
        if M == 0:
            return np.linspace(0, 1, 1000), np.repeat(w, 1000)
        X = np.linspace(0, 1, 1000)
        X = X.reshape((-1, 1))
        ori_X = X.copy()
        for i in range(2, M+1):
            X = np.concatenate((X, ori_X**i), axis=1)
        X = np.concatenate((np.ones((1000,1)), X), axis=1)
        Y = np.matmul(X, w)
        return np.linspace(0, 1, 1000), Y

if __name__ == '__main__':
    dataset = Dataset(1000)
    X, Y = dataset.gen_data()
    # sampling
    #sample_index = np.random.choice([i for i in range(X.shape[0])], 10, replace=False)
    sample_index = np.array([item for item in range(0, 1000, 100)])
    #sample_index = np.sort(sample_index)
    sample_X, sample_Y = X[sample_index], Y[sample_index]
    sample_X = sample_X.reshape((-1, 1))
    sample_Y = dataset.add_noise(sample_Y)
    curve = PolynomialCurveFitting()
    M = [3]
    for idx, m in enumerate(M):
        w = curve.fit(sample_X, sample_Y, m)
        pred_X, pred_Y = curve.gen_image(m, w)
        plt.subplot(2, 2, idx+1)
        plt.plot(X, Y, 'g')
        plt.plot(sample_X, sample_Y, 'bo')
        plt.plot(pred_X, pred_Y, 'r')
    plt.show()