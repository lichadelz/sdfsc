import fcl
import torch
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation
from .utils import rot_2d, euler2mat, DH2mat, rotz, wrap2pi
import trimesh

class Model():
    def __init__(self):
        self.dof = None
        self.limits = None
        
    def fkine(self, q):
        raise NotImplementedError
    
    def polygons(self, q):
        raise NotImplementedError
    
    def wrap(self, q):
        raise NotImplementedError


class DHParameters():
    def __init__(self, a=0, alpha=0, d=0, theta=0):
        self.a = torch.FloatTensor(a)
        self.alpha = torch.FloatTensor(alpha)
        self.d = torch.FloatTensor(d)
        self.theta = torch.FloatTensor(theta)
    
    def cuda(self):
        self.a = self.a.cuda()
        self.alpha = self.alpha.cuda()
        self.d = self.d.cuda()
        self.theta = self.theta.cuda()


class PandaFK(Model):
    def __init__(self):
        # measurement source: 
        # https://frankaemika.github.io/docs/control_parameters.html
        self.limits = torch.FloatTensor([[-2.8973, 2.8973],
                        [-1.7628, 1.7628],
                        [-2.8973, 2.8973],
                        [-3.0718, -0.0698],
                        [-2.8973, 2.8973],
                        [-0.0175, 3.7525],
                        [-2.8973, 2.8973]])
        L = torch.FloatTensor([
            0.3330, # L0
            0.3160, # L1
            0.0825, # L2, Between joint 3 and joint 4 
            0.3840, # L3
            0.0880, # L4
            0.1070*2, # L5
            ])
        self.L = L
        
        # modeling source: 
        # https://www.researchgate.net/publication/299640286_Baxter_Kinematic_Modeling_Validation_and_Reconfigurable_Representation
        self.dhparams = DHParameters(
            # a =   [   0,     0,    0, L[2], -L[2],    0, L[4]],
            # alpha=[   0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2],
            # d=    [L[0],     0, L[1],    0,  L[3],    0, L[5]],
            # theta=[   0,     0,    0,    0,     0,    0,    0]
            a =   [    0,    0, L[2], -L[2],    0, L[4],     0],
            alpha=[-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2,     0],
            d=    [ L[0],     0, L[1],    0,  L[3],    0, L[5]],
            theta=[    0,     0,    0,    0,     0,    0,    0]
        )
        # print(self.dhparams.a, self.dhparams.alpha, self.dhparams.d)
        self.c_alpha = self.dhparams.alpha.cos()
        self.s_alpha = self.dhparams.alpha.sin()
        self.dof = 7
        self.fk_mask = [True, False, True, True, True, False, True]
        self.fkine_backup = None
    
    def fkine(self, q, reuse=False):
        if reuse:
            return self.fkine_backup
        q = torch.reshape(q, (-1, self.dof))
        angles = q + self.dhparams.theta
        tfs = DH2mat(angles, self.dhparams.a, self.dhparams.d, self.s_alpha, self.c_alpha)
        assert tfs.shape == (len(q), self.dof, 4, 4)
        cum_tfs = []
        tmp_tf = tfs[:, 0]
        if self.fk_mask[0]:
            cum_tfs.append(tmp_tf)
        for i in range(1, self.dof):
            tmp_tf = torch.bmm(tmp_tf, tfs[:, i])
            if self.fk_mask[i]:
                cum_tfs.append(tmp_tf)
        self.fkine_backup = torch.stack([t[:, :3, 3] for t in cum_tfs], dim=1)
        # print("self.fkine_backup =",self.fkine_backup )
        return self.fkine_backup

class PointRobot1D(Model):
    def __init__(self, limits):
        # limits: (dof+1) x 2. Last dimension for time.
        self.limits = torch.FloatTensor(limits)
        self.dof = 1
        pass

    def fkine(self, q):
        '''
        Assume q is from [0, 1], d dimensions
        '''
        q = torch.reshape(q, (-1, self.dof))
        return q * (self.limits[:-1, 1] - self.limits[:-1, 0]) + self.limits[:-1, 0]
    
    def normalize(self, q):
        return (q-self.limits[:, 0]) / (self.limits[:, 1]-self.limits[:, 0])
    
    def unnormalize(self, q):
        return q * (self.limits[:, 1]-self.limits[:, 0]) + self.limits[:, 0]











