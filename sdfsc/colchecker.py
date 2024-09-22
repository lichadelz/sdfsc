import numpy as np
from matplotlib import pyplot as plt
import torch
from sklearn.svm import SVC
from scipy import ndimage
from scipy.interpolate import Rbf
from tqdm import tqdm
from time import time
import sys
import os
import pickle
import json
sys.path.append('/home/lichalab/SDFSC/sdfsc')
import Fktransform_fast as fk_fast
from Obstacles import Obstacle
from network import parallel_CombinedModel,parallel_mutilgpu_CombinedModel,parallel_cudastream_CombinedModel

class CollisionChecker():
    def __init__(self, obstacles):
        self.obstacles = obstacles
    
    def predict(self, point):
        return torch.any(torch.stack([obs.is_collision(point) for obs in self.obstacles], dim=1), dim=1)
    
    def line_collision(self, start, target, res=50):
        points = map(lambda i: start + (target - start)/res*i, range(res))
        return any(map(lambda p: self.is_collision(p), points))
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
class CustomSVM_cuda():
    def __init__(self):
        with open('model/svm/svm_model_params.json', 'r') as f:
             loaded_params = json.load(f)
        loaded_params['support_points'] = torch.tensor(loaded_params['support_points'], dtype=torch.float32, device='cuda')
        loaded_params['dual_coefs'] = torch.tensor(loaded_params['dual_coefs'], dtype=torch.float32, device='cuda')
        intercept = torch.tensor(loaded_params['intercept'], dtype=torch.float32, device='cuda')  
        gamma = torch.tensor(loaded_params['gamma'], dtype=torch.float32, device='cuda')  
        self.support_points = loaded_params['support_points']
        self.dual_coefs = loaded_params['dual_coefs']
        self.intercept = intercept
        self.gamma = gamma
    def decision_function(self, X_tensor):
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        n_samples = X_tensor.shape[0]
        decision_values = torch.zeros(n_samples, device='cuda')
        for i in range(n_samples):
            test_point = X_tensor[i].unsqueeze(0) 
            test_point=test_point.to(torch.float32)
            kernel_values = torch.exp(-self.gamma * torch.cdist(test_point, self.support_points).pow(2))
            decision_values[i] = (self.dual_coefs  * kernel_values).sum() + self.intercept
        
        return decision_values


class colchecker(CollisionChecker):
    def __init__(self,use_selfcol=True):
        #load nn model
        nn_directory = "model/nn_final" 
        model_list = []
        for i in range(7):
            PATH1 = os.path.join(nn_directory, 'train-seg'+str(i+1)+'.pt')
            model = torch.load(PATH1)
            model.eval()
            model_list.append(model)
        multi_model = parallel_CombinedModel(model_list).cuda()
        multi_model.eval()
        scripted_model = multi_model
        # load CustomSVM
        selfcolckecker=CustomSVM_cuda()
        #initilaze Segcdist
        self.model_list = model_list
        self.nn_model = scripted_model
        self.svm_model = selfcolckecker
        self.use_svm = use_selfcol
        self._fkine=True
        self._cuda = True
        self.points = None

    def get_points(self,points):
        if points is not None:
            self.points = torch.tensor(points, dtype=torch.float32, device='cuda')
        else:
            self.points = None
        return self.points
    def normalize(self,matrix_tensor):
        q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],dtype=torch.float32, device='cuda')
        q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],dtype=torch.float32, device='cuda')
        matrix_norm_tensor = (matrix_tensor-q_min)/(q_max-q_min)*2-1
        return matrix_norm_tensor
    def cal_svm_score(self,result_tensor):
        score=-1/(1+torch.exp(4.175*result_tensor+1.6))
        return score
    def svm_score(self,q_tensor):
        assert isinstance(q_tensor, torch.Tensor), "Input must be a PyTorch Tensor"
        if q_tensor.is_cuda:
            # print("q_tensor is on CUDA")
            pass
        else:
            print("q_tensor is on CPU")
        q_norm_tensor = self.normalize(q_tensor)
        svc_result_tensor=self.svm_model.decision_function(q_norm_tensor)
        svm_score_tensor=self.cal_svm_score(svc_result_tensor)
        return svm_score_tensor
    def svc_score(self,q_tensor):
        assert isinstance(q_tensor, torch.Tensor), "Input must be a PyTorch Tensor"
        if q_tensor.is_cuda:
            # print("q_tensor is on CUDA")
            pass
        else:
            print("q_tensor is on CPU")
        q_norm_tensor = self.normalize(q_tensor)
        svc_result_tensor=self.svm_model.decision_function(q_norm_tensor)
        return svc_result_tensor


    def nn_score(self,q_tensor,points_tensor):
        assert isinstance(q_tensor, torch.Tensor), "q must be a PyTorch Tensor"
        assert isinstance(points_tensor, torch.Tensor), "points must be a PyTorch Tensor"
        if q_tensor.is_cuda and points_tensor.is_cuda:
            # print("q_tensor and points_tensor is on CUDA")
            pass
        else:
            print("q_tensor and points_tensor is on CPU")
        input_tensor=fk_fast.fktransform_fast(q_tensor,points_tensor)

        nn_score_tensor = self.nn_model(input_tensor)
        return nn_score_tensor.min()
    def get_scores(self, queries,custom_points=None):
        """
        Compute combined scores for each query using nearest neighbor and SVM methods.
        Args:
            queries : q
            custom_points : points
        """
        if isinstance(queries, torch.Tensor):
            queries_tensor = queries.cuda().requires_grad_()
        else:
            queries_tensor = torch.tensor(queries, dtype=torch.float32, device='cuda').requires_grad_()
        if self.points is None:
            if custom_points is not None:
                self.points = torch.tensor(custom_points, dtype=torch.float32, device='cuda').requires_grad_()
            else:
                raise ValueError("Points must be provided either during initialization or when calling the function.")
        if queries_tensor.dim() == 1:
            queries_tensor = queries_tensor.unsqueeze(0)
        # Compute scores for each query and concatenate
        scores = torch.cat([self._compute_single_score(query) for query in queries_tensor]).unsqueeze(1)
        # print("scores=",scores)
        return scores

    def _compute_single_score(self, query):
        """Helper function to compute score for a single query."""
        dist = self.nn_score(query, self.points)
        if self.use_svm==False:
            return dist.view(-1)
        svm_score = self.svm_score(query)
        if self.use_svm==-1:
            return dist - svm_score
        return dist + svm_score
    def is_collision(self, q,point):
        return self.getscore(q,point) < 0.002
    
if __name__ == '__main__':
    checker=colchecker(use_selfcol=True)
    q_row= torch.tensor([[-0.2181,  0.1234, -0.3432, -2.0262,  0.1656,  2.2471,  0.2165],
    [-1.5,  0.5, -0.6, -1.0,  0.8,  1.5,  0.5]], dtype=torch.float32)
    checker.get_points([[0,0.5,0]])
    score=checker.get_scores(q_row)
    print(score)