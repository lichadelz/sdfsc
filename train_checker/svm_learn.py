from matplotlib import pyplot as plt
import torch
from sklearn.svm import SVC
from sklearn.svm import SVC
import numpy as np
import pickle
import json
class KernelFunc:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError('You need to define your own __call__ function.')
class RQKernel(KernelFunc):
    def __init__(self, gamma, p=2):
        self.gamma = gamma
        self.p = p

    def __call__(self, xs, x_primes):
        if xs.ndim == 1:
            xs = xs[np.newaxis, :]
        xs = xs[:, np.newaxis] # change to [1, len(x), channel]
        xs=xs.transpose((1,0,2))
        print(xs.shape)
        print(x_primes[np.newaxis, :] .shape)
        pair_diff_np = x_primes[np.newaxis, :] - xs
        pair_diff = torch.tensor(pair_diff_np, dtype=torch.float32)
        kvalues = (1/(1+self.gamma/self.p*torch.sum(pair_diff**2, dim=2))**self.p)

        if kvalues.shape[0] == 1:
            kvalues = kvalues.squeeze_(0)
       
        return kvalues

class mySVM:
    def __init__(self, X, Y, kernel_func='rbf', gamma=1.0):
        self.x = X
        self.y = Y
        self.kernel_func = RQKernel(gamma) if kernel_func=='rq' else kernel_func
        self.gamma = gamma
        # self.beta = beta
        self.gains = None  # 初始化 gains，具体值取决于您的实现
        self.support_points = None  # 假设 support_points 需要在训练后设置

    def train(self, max_iteration=1000):
        # self.svm = SVC(C=1e8, kernel=self.kernel_func, gamma=self.gamma, max_iter=max_iteration)
        self.svm = SVC(C=50, kernel=self.kernel_func, gamma=self.gamma, max_iter=50000)
        self.svm.fit(self.x, self.y)
        
        # 假设 self.support_points 需要在训练后设置为支持向量
        self.support_points = self.svm.support_vectors_
        
        # 初始化 gains，这里假设 gains 的大小与支持向量的数量相同
        self.gains = np.zeros(self.svm.dual_coef_.shape)
        
        # 更新 gains，这里假设您想根据对偶系数设置 gains
        self.gains = self.svm.dual_coef_.reshape(-1)
        
        self.intercept = self.svm.intercept_
        self.svm_gamma = self.gamma  # 使用传入的 gamma 参数
        
        print('SVM Gamma: {}'.format(self.svm_gamma))
        print('ACC: {}'.format(np.sum((self.svm.predict(self.x) < 0) == (self.y < 0)) / len(self.y)))
    def save(self):
        svm_model=pickle.dumps(self.svm)
        f_model=open('SVM_model/selfcol', "wb+")
        f_model.write(svm_model)
        f_model.close()
        model_params = {
            'support_points': self.support_points.tolist(),  # 将numpy数组转换为列表
            'dual_coefs': self.gains.tolist(),
            'intercept': self.intercept.tolist(),
            'gamma': self.gamma,
        }
        with open('SVM_model/svm_model_params.json', 'w') as f:
            json.dump(model_params, f, indent=4)
        print ("Save Done\n")
    def predict(self,input):
        predict_y=self.svm.predict(input) 
        return predict_y
    def getscore(self,input):
        score=self.svm.decision_function(input) 
        return score
