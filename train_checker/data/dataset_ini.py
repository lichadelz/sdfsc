# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
import torch
import math
pi = math.pi

def dataset_selfcol_q(type='train'):
    """
    自碰撞检测关节角矩阵(归一化[-1,1])
    """
    if type=='train':
        # fmatrix_path = 'data/self_col''.txt'  # 指定 txt 文件的路径
        # filename=[2,3,4]
        filename=[2,4]
    elif type=='test':
        # fmatrix_path = 'data/self_col_test.txt'  # 指定 txt 文件的路径
        filename=[1]
    matrix_self_result=np.empty((0, 7))
    for i in filename:
        fmatrix_path = 'data/icra_data_self/matrix_self_train'+str(i)+'.txt'  # 指定 txt 文件的路径
        # 读取 txt 文件
        with open(fmatrix_path, 'r') as file:
            lines = file.readlines()
        # 提取文件中的数值并将其转换成矩阵
        matrix_self = []
        for line in lines:
            row = line.strip().split()  # 分割每行中的数值
            row = [float(num) for num in row]  # 将数值转换为浮点数
            matrix_self.append(row)
        matrix_self = np.array(matrix_self)  # 转换为 NumPy 数组
        print()
        matrix_self_result=np.vstack((matrix_self_result,matrix_self))
    return matrix_self_result
def dataset_selfcol(type='train'):
    """
    自碰撞检测结果
    """
    if type=='train':
        # fmatrix_path = 'data/self_col''.txt'  # 指定 txt 文件的路径
        # filename=[2,3,4]
        filename=[2,4]
    elif type=='test':
        # fmatrix_path = 'data/self_col_test.txt'  # 指定 txt 文件的路径
        filename=[1]
    # 读取 txt 文件
    self_col_result=np.empty((0, 1))
    for i in filename:
        print("filename:",i)
        fmatrix_path = 'data/icra_data_self/self_col_train'+str(i)+'.txt'  # 指定 txt 文件的路径
        with open(fmatrix_path, 'r') as file:
            lines = file.readlines()
        # 提取文件中的数值并将其转换成矩阵
        self_col = []
        for line in lines:
            row = line.strip().split()  # 分割每行中的数值
            row = [float(num) for num in row]  # 将数值转换为浮点数
            self_col.append(row)
        self_col = np.array(self_col)  # 转换为 NumPy 数组
        self_col_result=np.vstack((self_col_result,self_col.T))
    return self_col_result
def dataset_q():
    """
    真实关节角矩阵（非归一化）
    """
    fmatrix_path = 'data/matrix.txt'  # 指定 txt 文件的路径
    # 读取 txt 文件
    with open(fmatrix_path, 'r') as file:
        lines = file.readlines()
    # 提取文件中的数值并将其转换成矩阵
    matrix = []
    for line in lines:
        row = line.strip().split()  # 分割每行中的数值
        row = [float(num) for num in row]  # 将数值转换为浮点数
        matrix.append(row)
    matrix = np.array(matrix)  # 转换为 NumPy 数组
    q = q_unnorm(matrix)
    return q
    

def dataset_pos():
    """
    点位置矩阵
    """
    fpos_path = 'data/pos.txt'  # 指定 txt 文件的路径
    # 读取 txt 文件
    with open(fpos_path, 'r') as file:
        lines = file.readlines()
    # 提取文件中的数值并将其转换成矩阵6
    pos = []
    for line in lines:
        row = line.strip().split()  # 分割每行中的数值
        row = [float(num) for num in row]  # 将数值转换为浮点数
        pos.append(row)
    pos = np.array(pos)  # 转换为 NumPy 数组
    return pos

def dataset_cdist():
    """
    关节距离矩阵
    """
    fpos_path = 'data/result_dist.txt'  # 指定 txt 文件的路径
    # 读取 txt 文件
    with open(fpos_path, 'r') as file:
        lines = file.readlines()
    # 提取文件中的数值并将其转换成矩阵6
    cdist = []
    for line in lines:
        row = line.strip().split()  # 分割每行中的数值
        row = [float(num) for num in row]  # 将数值转换为浮点数
        cdist.append(row)
    cdist = np.array(cdist)  # 转换为 NumPy 数组
    return cdist

def dataset_min_joint():
    """
    距离最小关节矩阵
    """
    fpos_path = 'data/min_joint_gjk.txt'  # 指定 txt 文件的路径
    # 读取 txt 文件
    with open(fpos_path, 'r') as file:
        lines = file.readlines()
    # 提取文件中的数值并将其转换成矩阵
    min_joint = []
    for line in lines:
        row = line.strip().split()  # 分割每行中的数值
        row = [float(num) for num in row]  # 将数值转换为浮点数
        min_joint.append(row)
    min_joint = np.array(min_joint)  # 转换为 NumPy 数组
    return min_joint


def dataset_result(gjk=0):
    """
    fcl或gjk距离矩阵
    """
    if gjk==0:
        fpos_path = 'data/result_fcl.txt'  # 指定 txt 文件的路径
    else:
        fpos_path = 'data/result_gjk.txt'  # 指定 txt 文件的路径
    # 读取 txt 文件
    with open(fpos_path, 'r') as file:
        lines = file.readlines()
    # 提取文件中的数值并将其转换成矩阵
    result = []
    for line in lines:
        row = line.strip().split()  # 分割每行中的数值
        row = [float(num) for num in row]  # 将数值转换为浮点数
        result.append(row)
    result = np.array(result)  # 转换为 NumPy 数组
    return result

def q_norm(matrix):
    """
    真实q->归一化q
    """
    if isinstance(matrix, torch.Tensor):
        # 将PyTorch张量转换为NumPy数组
        matrix= matrix.detach().numpy()
    else:
        # 如果不是torch.Tensor，直接返回原变量
        pass
    q_min = np.array([-2.8973, -1.7628, -2.8973, -
                     3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q_norm = (matrix-q_min)/(q_max-q_min)*2-1
    return q_norm
def q_unnorm(matrix):
    """
    归一化q->真实q
    """
    q_min = np.array([-2.8973, -1.7628, -2.8973, -
                     3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    q_unnorm = (matrix+1.0)/2*(q_max-q_min)+q_min
    return q_unnorm
