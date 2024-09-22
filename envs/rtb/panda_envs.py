from .. import CollisionEnv

import roboticstoolbox as rtb
from roboticstoolbox.models import Panda
from spatialgeometry import Cuboid, Cylinder, Sphere, Mesh
import spatialgeometry as sg
import spatialmath as sm
from spatialmath import SE3
from swift import Swift

# Path: envs/rtb/panda_envs.py

class PandaEnv(CollisionEnv):
    '''
    General collision environment for Panda robot.
    Add your own objects to create a custom environment.

    Objects: dict[key: (shape type, other shape parameters[dict])]
    '''
    def __init__(self, object_info: dict=None, launch_args: dict=None):
        super().__init__()
        self.robot = Panda()
        self.robot.q = self.robot.qr
        self.env = self._launch_env(launch_args)
        self._add_objects(object_info)

    
    def _launch_env(self, launch_args: dict):
        '''
        Launch the collision environment.

        Parameters:
            launch_args: dict
        '''
        if launch_args is None:
            launch_args = {}
        env = Swift()
        env.launch(**launch_args)
        env.add(self.robot)
        #background
        s_ground=sg.Cuboid(scale=[50,50,0.00001],base=sm.SE3(0,0,0.001),color=[0.5,0.5,0.5])
        s1_ground=sg.Cuboid(scale=[0.0001,50,50],base=sm.SE3(24,0,0.01),color=[1,1,1])
        s2_ground=sg.Cuboid(scale=[0.0001,50,50],base=sm.SE3(-24,0,0.01),color=[1,1,1])
        s3_ground=sg.Cuboid(scale=[50,0.0001,50],base=sm.SE3(0,24,0.01),color=[1,1,1])
        s4_ground=sg.Cuboid(scale=[50,0.0001,50],base=sm.SE3(0,-24,0.01),color=[1,1,1])
        table_ground=sg.Cuboid(scale=[0.8,1.6,0.02],base=sm.SE3(0.5,0,0.24),color=[0.804,0.5215,0.241])
        table_leg1=sg.Cuboid(scale=[0.08,0.08,0.24],base=sm.SE3(0.2,0.6,0.12),color=[0.804,0.5215,0.241])
        table_leg2=sg.Cuboid(scale=[0.08,0.08,0.24],base=sm.SE3(0.2,-0.6,0.12),color=[0.804,0.5215,0.241])
        table_leg3=sg.Cuboid(scale=[0.08,0.08,0.24],base=sm.SE3(0.8,0.6,0.12),color=[0.804,0.5215,0.241])
        table_leg4=sg.Cuboid(scale=[0.08,0.08,0.24],base=sm.SE3(0.8,-0.6,0.12),color=[0.804,0.5215,0.241])
        env.add(s_ground)
        env.add(s1_ground)
        env.add(s2_ground)
        env.add(s3_ground)
        env.add(s4_ground)
        env.add(table_leg1)
        env.add(table_leg2)
        env.add(table_leg3)
        env.add(table_leg4)
        env.add(table_ground)
        return env
    
    def _add_objects(self, object_info: dict):
        '''
        Add objects to the environment.

        Parameters:
            objects: dict[shape type: shape parameters[dict]]
        '''
        self.objects = {}
        shape_class_map = {
            'box': Cuboid,
            'cuboid': Cuboid,
            'cylinder': Cylinder,
            'sphere': Sphere,
            'mesh': Mesh
        }
        for shape_key, (shape_type, shape_params) in object_info.items():
            if shape_type in shape_class_map:
                shape_class = shape_class_map[shape_type]
                shape_obj = shape_class(**shape_params)
                self.env.add(shape_obj)
                self.objects[shape_key] = shape_obj
            else:
                raise NotImplementedError
    
    def _single_collision(self, q):
        collided = [self.robot.iscollided(q, obj) for _, obj in self.objects.items()]
        return any(collided)
    
    def _single_distance(self, q):
        dists = [self.robot.closest_point(q, obj)[0] for _, obj in self.objects.items()]
        return min(dists)
    
    def sample_q(self):
        return self.robot.random_q()


class PandaSingleCylinderEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cylinder.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cylinder1': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(10, 10.0, 0.4),
                'color': (1.0, 1.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)

class PandaThreeCylinderEnv(PandaEnv):
    '''
    Collision environment for Panda robot with three cylinders.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cylinder1': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.3, -0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cylinder2': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cylinder3': ('cylinder', {
                'radius': 0.05,
                'length': 0.8,
                'pose': SE3(0.3, 0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)

class PandaCuboidEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cuboid.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cuboid1': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.3, -0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid2': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid3': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.3, 0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)
class PandaFourCuboidEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cuboid.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cuboid1': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.3, -0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid2': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid3': ('cuboid', {
                'scale': [0.15, 0.15, 0.3],
                'pose': SE3(0.3, 0.5, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid4': ('cuboid', {
                'scale': [0.1, 0.1, 0.45],
                'pose': SE3(0.25, 0.0, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            })
        }
        super().__init__(object_info, launch_args)

class PandamultiCuboidEnv(PandaEnv):
    '''
    Collision environment for Panda robot with a single cuboid.
    '''
    def __init__(self, launch_args: dict=None):
        object_info = {
            'cuboid1': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.3, -0.5, 0.5),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid2': ('cuboid', {
                'scale': [0.5, 0.05, 0.4],
                'pose': SE3(0.5, 0, 0.4),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid3': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.2, 0.4, 0.45),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid4': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.55, 0.1, 0.65),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid5': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.3, -0., 0.75),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid6': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.45, -0.1, 0.65),
                'color': (1.0, 0.0, 0.0, 1.)
            }),
            'cuboid7': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(-0.2, 0.2, 0.65),
                'color': (1.0, 0.0, 0.0, 1.)
            })
            ,
            'cuboid8': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.4, 0.3, 0.75),
                'color': (1.0, 0.0, 0.0, 1.)
            })
            ,
            'cuboid9': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(0.1, -0.4, 0.65),
                'color': (1.0, 0.0, 0.0, 1.)
            })
            ,
            'cuboid10': ('cuboid', {
                'scale': [0.1, 0.1, 0.1],
                'pose': SE3(-0.1, -0.3, 0.65),
                'color': (1.0, 0.0, 0.0, 1.)
            })
            
            
            
        }
        super().__init__(object_info, launch_args)