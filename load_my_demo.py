import pickle
import numpy as np
from os import listdir
from os.path import join, exists

from typing import List


from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from helpers import utils
from natsort import natsorted

import numpy as np
from scipy.spatial.transform import Rotation as R

def get_demos(path, task, amount, resolution, random_selection: bool = False, from_episode_number: int=0):
    #TODO: fill "join" function with correct path
    examples_path = join(path, task)
    examples = listdir(examples_path)

    if amount == -1:
        amount = len(examples)

    if amount > len(examples):
        raise RuntimeError('You asked for %d examples, but only %d were available.' % (amount, len(examples)))

    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(examples)[from_episode_number:from_episode_number+amount]

    Rot_M = np.array([
            [-0.56396701,  0.24796995, -0.78768783],
            [ 0.42248543,  0.90620631, -0.01720975],
            [ 0.70954019, -0.34249237, -0.61583415]], dtype=np.float32)





    tvec = np.array([[ 0.0932756,  -0.168814,   -1.01014076]], dtype=np.float32)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)

        print(example)
        #with open(join(example_path, 'obs.pkl'), 'rb') as f:
        with open(example_path, 'rb') as f:
            obs = pickle.load(f)

        num_steps = len(obs)
        rollout = []
        for t in range(num_steps):
            
            obs_t = Observation(
                    left_shoulder_rgb=None,
                    left_shoulder_depth=None,
                    left_shoulder_point_cloud=None,
                    right_shoulder_rgb=None,
                    right_shoulder_depth=None,
                    right_shoulder_point_cloud=None,
                    overhead_rgb=None,
                    overhead_depth=None,
                    overhead_point_cloud=None,
                    wrist_rgb=None, #obs[t]['realsense/image'][:128, :128, :],
                    wrist_depth=None, #obs[t]['realsense/depth'],
                    wrist_point_cloud=None, #obs[t]['realsense/point_cloud'][:128, :128, :],
                    front_rgb=obs[t]['image'],
                    #front_rgb=obs[t]['kinect/image'][::4, ::5, :][8:136, :, :],

                    #TODO Rotations and translations
                    front_depth=None, #obs[t]['kinect/depth'],

                    
                    front_point_cloud=obs[t]['point_cloud'],
                    #front_point_cloud=np.matmul((obs[t]['kinect/point_cloud'][::5, ::10, :][8:136, :, :] + tvec),Rot_M),
                    #front_point_cloud=obs[t]['kinect/point_cloud'][::4, ::5, :][8:136, :, :],
                    


                    left_shoulder_mask=None,
                    right_shoulder_mask=None,
                    overhead_mask=None,
                    wrist_mask=None,
                    front_mask=None,
                    joint_velocities=obs[t]['joint_velocity'],
                    joint_positions=None,
                    joint_forces=None,
                    gripper_open=1 if obs[t]['gripper_pos'] < 0.1 else 0,
                #(1.0 if self.robot.gripper.get_open_amount()[0] > 0.95 else 0.0) # Changed from 0.9 to 0.95 because objects, the gripper does not close completely
                    gripper_pose=np.concatenate((obs[t]['tcp_pose'][:3], obs[t]['tcp_pose'][3:])),
                    #gripper_pose=obs[t]['arm/ActualTCPPose'],


                    #gripper_pose=np.concatenate(((obs[t]['arm/ActualTCPPose'][:3] + np.array([0.0 , 0.0, 0.0469], dtype=np.float16)), utils.discrete_euler_to_quaternion(obs[t]['arm/ActualTCPPose'][3:], resolution))),
                    gripper_matrix=None,
                    gripper_touch_forces=None,
                    gripper_joint_positions=[obs[t]['gripper_pos'], obs[t]['gripper_pos']],
                    task_low_dim_state=None,
                    ignore_collisions=0.0,
                    misc={'descriptions': [str(obs[t]['description'])],
                          #['Put red santa in the box'],
                          'gripper_object_detected': [obs[t]['gripper_is_obj_detected']],
                          'front_camera_extrinsics': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                          #'front_camera_extrinsics': np.array([
            #[-0.56396701,  0.24796995, -0.78768783, 0.0932756],
            #[ 0.42248543,  0.90620631, -0.01720975, -0.168814],
            #[ 0.70954019, -0.34249237, -0.61583415, -1.01014076],
            #                               [0, 0, 0, 1]], dtype=np.float32),
                          'front_camera_intrinsics': [[1,0,0],[0,1,0],[0,0,1]],
                          'wrist_camera_extrinsics': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                          'wrist_camera_intrinsics': [[1,0,0],[0,1,0],[0,0,1]]})

            rollout.append(obs_t)

        #print(example)
        demo = Demo(rollout)
        demos.append(demo)
    return demos
