import torch
import fcl
import numpy as np
from sklearn.svm import SVC
from scipy import ndimage
from scipy.interpolate import Rbf
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
class Obstacle:
    def __init__(self, kind, position, size, num_samples=100,cost=np.inf):
        self.kind = kind
        if self.kind not in ['cylinder', 'cuboid','sphere']:
            raise NotImplementedError('Obstacle kind {} not supported'.format(kind))
        self.position = torch.FloatTensor(position)
        self.size = torch.FloatTensor(size) 
        self.cost = cost
        self.points = self.get_points(num_samples)
    """
    Determine whether the point is within the obstacle
    """
    def is_collision(self, point):
        if point.ndim == 1:
            point = point[np.newaxis, :]
        if self.kind == 'cylinder':
            # Distance along the z-axis (assuming cylinder axis is aligned with z-axis)
            dist_z = torch.abs(point[:, 2] - self.position[2])
            # Distance in the xy-plane
            dist_xy = torch.sqrt((point[:, 0] - self.position[0])**2 + (point[:, 1] - self.position[1])**2)
            return (dist_z < self.size[1]/2) & (dist_xy < self.size[0]/2)
        elif self.kind == 'cuboid':
            return torch.all(torch.abs(self.position-point) < self.size/2, dim=1)
        elif self.kind == 'sphere':
            # Euclidean distance between point and sphere center
            dist = torch.norm(self.position-point, dim=1)
            return dist <= self.size
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    
    def get_cost(self):
        return self.cost
    """
    Sampling on the surface of obstacles, uniformly sampling according to the surface area
    """
    def get_points(self,num_samples):
        if self.kind == 'cylinder':
            return self.get_cylinder_points(num_samples)
        elif self.kind == 'cuboid':
            return self.get_cuboid_points(num_samples)
        elif self.kind == 'sphere':
            return self.get_sphere_points(num_samples)
        else:
            raise NotImplementedError('Obstacle kind {} not supported'.format(self.kind))
    def get_cylinder_points(self,num_samples):
        radius, height = self.size
        samples = torch.zeros((num_samples, 3))
        face_weights = {'top': np.pi * radius**2 , 'bottom': np.pi * radius**2 , 'side': 2 * np.pi * radius * height }
        total_weight = sum(face_weights.values())
        face_probabilities = {face: weight / total_weight for face, weight in face_weights.items()}
        print(face_probabilities)

        for i in range(num_samples):
            face = np.random.choice(list(face_probabilities.keys()), p=np.array(list(face_probabilities.values())))
            if face == 'top' or face == 'bottom':
                # Top and bottom faces
                r=radius * torch.rand(1)
                angle = 2 * torch.pi * torch.rand(1)
                x = self.position[0] + r * torch.cos(angle)
                y = self.position[1] + r * torch.sin(angle)
                z = self.position[2] + height*0.5 if face == 'top' else self.position[2]- height*0.5 
            elif face == 'side':
                # Side faces
                angle =  2 * torch.pi * torch.rand(1)
                x = self.position[0] + radius * torch.cos(angle)
                y = self.position[1] + radius * torch.sin(angle)
                z = self.position[2] + height * (torch.rand(1) - 0.5)
            samples[i] = torch.tensor([x, y, z])
        return samples
    def get_cuboid_points(self,num_samples):
        width, height, depth = self.size
        samples = torch.zeros((num_samples, 3))
        face_weights = {'top': width*height, 'bottom': width*height, 'front': width * depth, 'back': width * depth, 'left': height * depth, 'right': height * depth}
        total_weight = sum(face_weights.values())
        face_probabilities = {face: weight / total_weight for face, weight in face_weights.items()}
        print(face_probabilities)
        for i in range(num_samples):
            # Select a face based on its area-weighted probability
            face = np.random.choice(list(face_probabilities.keys()), p=np.array(list(face_probabilities.values())))
            if face == 'top' or face == 'bottom':
                # Top and bottom faces
                x = self.position[0] + width * (torch.rand(1) - 0.5)
                y = self.position[1] + height * (torch.rand(1) - 0.5)
                z = self.position[2] + depth*0.5 if face == 'top' else self.position[2]- depth*0.5 
            elif face == 'front' or face == 'back':
                # Front and back faces
                x = self.position[0] + width * (torch.rand(1) - 0.5)
                z = self.position[2] + depth * (torch.rand(1) - 0.5)
                y = self.position[1] + height*0.5 if face == 'front' else self.position[1]-height*0.5 
            else:  # 'left' or 'right'
                # Left and right faces
                y = self.position[1] + height * (torch.rand(1) - 0.5)
                z = self.position[2] + depth * (torch.rand(1) - 0.5)
                x = self.position[0] + width*0.5 if face == 'right' else self.position[0]- width*0.5

            samples[i] = torch.tensor([x, y, z])
        return samples
    def get_sphere_points(self, num_samples):
        radius = self.size  # Assuming the size attribute is the radius
        samples = torch.zeros((num_samples, 3))

        for i in range(num_samples):
            # Generate random latitude and longitude angles
            phi = torch.acos(1 - 2 * torch.rand(1))  # [-π/2, π/2]
            theta = 2 * torch.pi * torch.rand(1)  # [0, 2π]

            # Convert spherical coordinates to Cartesian coordinates
            x = self.position[0] +radius * torch.sin(phi) * torch.cos(theta)
            y = self.position[1] +radius * torch.sin(phi) * torch.sin(theta)
            z = self.position[2] +radius * torch.cos(phi)

            samples[i] = torch.tensor([x, y, z])

        return samples
    def move(self, v):
        self.position += v
        self.points += v
        return self.points
if __name__ == '__main__':
    obstacles = [
        ('cylinder', [0.3, -0.5, 0.4], [0.05,0.8]),
        ('cuboid', [0, 0,0.2], [0.2, 0.2,0.2]),
        ('sphere', [0, 0, 0.5], [0.5])
    ]
    obstacles = [Obstacle(*obstacle) for obstacle in obstacles]
    points=obstacles[0].points
    points = points.numpy()
    print(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.6, 0.6)  
    ax.set_ylim(-0.6, 0.6)  
    ax.set_zlim(0, 1) 

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()