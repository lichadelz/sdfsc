# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import torch
pi = torch.pi
num_vector = 7

def fktransform_fast(q_row,points):
    d = torch.tensor([0.333, 0, 0.316,0,       0.384,  0, 0], dtype=torch.float32, device='cuda')
    a = torch.tensor([0,     0, 0,    0.0825, -0.0825, 0, 0.088], dtype=torch.float32, device='cuda')
    T = torch.eye(4, dtype=torch.float32, device='cuda')
    Ti_all = torch.zeros((7, 4, 4), dtype=torch.float32, device='cuda')
    zero=torch.tensor(0, dtype=torch.float32, device='cuda')
    one=torch.tensor(1, dtype=torch.float32, device='cuda')

    s0 = torch.sin(q_row[0])
    c0 = torch.cos(q_row[0])
    row01 = torch.stack([c0, -s0, zero, zero]).unsqueeze(0)
    row02 = torch.stack([s0, c0, zero, zero]).unsqueeze(0)
    row03 = torch.stack([zero, zero, one, d[0]]).unsqueeze(0)
    row_last = torch.stack([zero, zero, zero, one]).unsqueeze(0)
    T0i = torch.cat([row01, row02, row03, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T0i)
    Ti_all[0, :, :] = T
    # print("FAST T0=",T)
    s1 = torch.sin(q_row[1])
    c1 = torch.cos(q_row[1])
    row11 = torch.stack([c1, -s1, zero, zero]).unsqueeze(0)
    row12 = torch.stack([zero, zero, one, zero]).unsqueeze(0)
    row13 = torch.stack([-s1, -c1, zero, zero]).unsqueeze(0)
    T1i = torch.cat([row11, row12, row13, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T1i)
    Ti_all[1, :, :] = T
    # print("FAST T1=",T)
    s2 = torch.sin(q_row[2])
    c2 = torch.cos(q_row[2])
    row21 = torch.stack([c2, -s2, zero, zero]).unsqueeze(0)
    row22 = torch.stack([zero, zero, -one, -d[2]]).unsqueeze(0)
    row23 = torch.stack([s2, c2, zero, zero]).unsqueeze(0)
    T2i = torch.cat([row21, row22, row23, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T2i)
    Ti_all[2, :, :] = T
    # print("FAST T2=",T)
    s3 = torch.sin(q_row[3])
    c3 = torch.cos(q_row[3])
    row31 = torch.stack([c3, -s3, zero, a[3]]).unsqueeze(0)
    row32 = torch.stack([zero, zero, -one, zero]).unsqueeze(0)
    row33 = torch.stack([s3, c3, zero, zero]).unsqueeze(0)
    T3i = torch.cat([row31, row32, row33, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T3i)
    Ti_all[3, :, :] = T
    # print("FAST T3=",T)
    s4 = torch.sin(q_row[4])
    c4 = torch.cos(q_row[4])
    row41 = torch.stack([c4, -s4, zero, a[4]]).unsqueeze(0)
    row42 = torch.stack([zero, zero, one, d[4]]).unsqueeze(0)
    row43 = torch.stack([-s4, -c4, zero, zero]).unsqueeze(0)
    T4i = torch.cat([row41, row42, row43, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T4i)
    Ti_all[4, :, :] = T
    # print("FAST T4=",T)
    s5 = torch.sin(q_row[5])
    c5 = torch.cos(q_row[5])
    row51 = torch.stack([c5, -s5, zero, zero]).unsqueeze(0)
    row52 = torch.stack([zero, zero, -one, zero]).unsqueeze(0)
    row53 = torch.stack([s5, c5, zero, zero]).unsqueeze(0)
    T5i = torch.cat([row51, row52, row53, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T5i)
    Ti_all[5, :, :] = T
    # print("FAST T5=",T)
    s6 = torch.sin(q_row[6])
    c6 = torch.cos(q_row[6])
    row61 = torch.stack([c6, -s6, zero, a[6]]).unsqueeze(0)
    row62 = torch.stack([zero, zero, -one, zero]).unsqueeze(0)
    row63 = torch.stack([s6, c6, zero, zero]).unsqueeze(0)
    T6i = torch.cat([row61, row62, row63, row_last], dim=0).to(torch.float32)
    T = torch.matmul(T, T6i)
    Ti_all[6, :, :] = T
    # print("FAST T6=",T)
    inv_Ti = torch.inverse(Ti_all).cuda()
    T0p = torch.eye(4).unsqueeze(0).repeat(points.shape[0], 1, 1).cuda()
    T0p[:, :3, 3] = points
    T1p = torch.matmul(inv_Ti.unsqueeze(1), T0p)
    pos_i = T1p[:, :, :3, 3]
    return pos_i