from sdfsc.Obstacles import Obstacle
import matplotlib.pyplot as plt
import torch
# obstacles = [
#     ('cylinder', [0.3, -0.5, 0.4], [0.05,0.8]),
#     ('cylinder', [0.5,0, 0.4], [0.05,0.8]),
#     ('cylinder', [0.3, 0.5, 0.4], [0.05,0.8])
# ]

obstacles = [
    ('cuboid', [0.3, -0.5, 0.5], [0.1, 0.1, 0.1]),
    ('cuboid', [0.5, 0, 0.4], [0.5, 0.05, 0.4]),
    ('cuboid', [0.2, 0.4, 0.45], [0.1, 0.1, 0.1]),
    ('cuboid', [0.55, 0.1, 0.65], [0.1, 0.1, 0.1]),
    ('cuboid', [0.3, -0., 0.75], [0.1, 0.1, 0.1]),
    ('cuboid', [0.45, -0.1, 0.65], [0.1, 0.1, 0.1]),
    ('cuboid', [-0.2, 0.2, 0.65], [0.1, 0.1, 0.1]),
    ('cuboid', [0.4, 0.3, 0.75], [0.1, 0.1, 0.1]),
    ('cuboid', [0.1, -0.4, 0.65], [0.1, 0.1, 0.1]),
    ('cuboid', [-0.1, -0.3, 0.6], [0.1, 0.1, 0.1]),

]
obstacles = [Obstacle(*obstacle) for obstacle in obstacles]
points_list = []

for i in range(len(obstacles)):

    points = obstacles[i].points

    points_list.append(points)

all_points = torch.cat(points_list, dim=0)

print(all_points.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.6, 0.6)  
ax.set_ylim(-0.6, 0.6)  
ax.set_zlim(0, 1)  

ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2])

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')

plt.show()