import os
import numpy as np
import data.dataset_ini as ini
from svm_learn import mySVM
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
def train():
    selfcol_q=ini.dataset_selfcol_q()
    selfcol_result=ini.dataset_selfcol()
    print(selfcol_q.shape)
    print(selfcol_q[:5])
    print(selfcol_result.shape)
    print(selfcol_result[:5])
    selfcolckecker=mySVM(selfcol_q,selfcol_result.flatten(),'rbf')
    selfcolckecker.train(max_iteration=50000)
    selfcolckecker.save()

def test():
    f2=open('SVM_model/selfcol','rb')
    svm_model=f2.read()
    selfcolckecker=pickle.loads(svm_model)
    selfcol_q_test=ini.dataset_selfcol_q('test')
    selfcol_result_test=ini.dataset_selfcol('test').flatten()
    # selfcol_q_test=ini.dataset_selfcol_q('train')
    # selfcol_result_test=ini.dataset_selfcol('train').flatten()
    print(selfcol_result_test.shape)
    pre_y=selfcolckecker.decision_function(selfcol_q_test)
    print(selfcol_q_test.shape)
    print(pre_y[:20])
    mean_y = np.mean(pre_y)
    std_dev_y = np.std(pre_y)
    print(f'Mean: {mean_y}')
    print(f'Standard Deviation: {std_dev_y}') 
    print(mean_y-std_dev_y)
    plt.figure(0)
    sns.kdeplot(pre_y, fill=True,label='SVM Score')

    same_sign_indices = np.where((pre_y >= 0) == (selfcol_result_test >= 0))
    pre_y_same = pre_y[same_sign_indices]
    mean_same = np.mean(pre_y_same)
    std_dev_same = np.std(pre_y_same)
    print(f'Mean_same: {mean_same}')
    print(f'Standard Deviation same: {std_dev_same}')
    pre_y_same_p = np.empty_like(pre_y)
    pre_y_same_p.fill(40)
    pre_y_same_p[same_sign_indices] = pre_y[same_sign_indices]
    sns.kdeplot(pre_y_same_p, fill=True, label='Correctly Classified')

    different_sign_indices = np.where((pre_y >= 0) != (selfcol_result_test >= 0))
    pre_y_diff= pre_y[different_sign_indices]
    mean = np.mean(pre_y_diff)
    std_dev = np.std(pre_y_diff)
    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std_dev}') 
    print(mean-std_dev)
    pre_y_diff_p = np.empty_like(pre_y)
    pre_y_diff_p.fill(40)
    pre_y_diff_p[different_sign_indices] = pre_y[different_sign_indices]
    sns.kdeplot(pre_y_diff_p, fill=True, label='Misclassified')

    plt.xlim(-15, 25)
    plt.ylim(0, 0.08)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig('selfcol_test.png',dpi=800)
    plt.show()
    threshold=-0.37
    t0 = time.time()
    print('ACC: {}'.format(np.sum((pre_y< threshold) == (selfcol_result_test< 0)) / len(selfcol_result_test)))
    t1 = time.time()
    print("Time: %f" % (t1 - t0))
    #TPR
    negative_pre_y_indices = pre_y < threshold
    negative_at_same_indices = (selfcol_result_test[negative_pre_y_indices] < 0)
    probability = np.mean(negative_at_same_indices)
    print(f"TPR: {probability}") 
    #TNR
    negative_pre_y_indices = pre_y > threshold
    negative_at_same_indices = (selfcol_result_test[negative_pre_y_indices] > 0)
    probability = np.mean(negative_at_same_indices)
    print(f"TNR: {probability}") 
if __name__ == '__main__':
    # train()
    test()