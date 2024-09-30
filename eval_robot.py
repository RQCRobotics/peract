import numpy as np

import pickle
from helpers import utils 
from ur_env.remote import RemoteEnvServer, RemoteEnvClient
from agents.peract_bc import launch_utils

import random

from helpers.clip.core.clip import tokenize

from scipy.spatial.transform import Rotation as R

import torch
import copy


from helpers import utils


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pprint
from rlbench.backend.observation import Observation


#tokens = ['Put the red santa in the box',
#          'Put the pink cat in the box',
#          'Put the grey seal in the box',
#          'Put the white cat in the box'
#          ]

#tokens = ['Open the top drawer',
#          'Open the midle drawer',
#          ]


tokens = ['Put the red toy in the box',
          'Put the white toy in the box',
          ]


#tokens = ['Put the red toy in the box',
#          'Put the green toy in the box',
#          'Put the blue toy in the box',
#          'Put the yellow toy in the box',
#          ]




#tokens = ['Place the green cube on top of the blue cube',
#          'Place the green cube on top of the yellow cube',
#          'Place the green cube on top of the red cube',
#          'Place the red cube on top of the blue cube',
#          'Place the red cube on top of the yellow cube',
#          'Place the red cube on top of the green cube',
#          'Place the yellow cube on top of the blue cube',
#          'Place the yellow cube on top of the red cube',
#          'Place the yellow cube on top of the green cube',
#          'Place the blue cube on top of the yellow cube',
#          'Place the blue cube on top of the red cube',
#          'Place the blue cube on top of the green cube',
#          ]


random.seed(123)



class UR5_env:
    def __init__(self,
                 cfg,
                 weights_path=None,
                 host='192.168.1.136',
                 port=5555,
                 #host='0.tcp.eu.ngrok.io',
                 #port=19512
                 ):
        self.device = torch.device('cuda:0')
        self.weights_path = weights_path
        if self.weights_path == None:
            raise 'No weights path as argument'
        self.cfg = cfg

        self.port = port
        self.host = host
        self.address = (host, port)
        import pdb;pdb.set_trace()
        self.client = RemoteEnvClient(self.address)
        self.step_n = 0
        self.observation = None
        self.lang_goal_t = None
        self.Rot_M = np.array([
            [-0.56396701,  0.24796995, -0.78768783],
            [ 0.42248543,  0.90620631, -0.01720975],
            [ 0.70954019, -0.34249237, -0.61583415]], dtype=np.float32)

       
        self.tvec = np.array([[ 0.0932756,  -0.168814,   -1.01014076]], dtype=np.float32)
        self._t = 0


    def _load_agent(self):
        self.agent = launch_utils.create_agent(self.cfg)
        self.agent.build(training=False, device=self.device)
        self.agent.load_weights(self.weights_path)
        print("Loaded: " + self.weights_path)

    def launch(self):
        self._load_agent()
        self.observation = self.reset()
        print(self.observation['tcp_pose'])
        #self.observation['arm/ActualTCPPose'] -= 0.0469
        return True

    def reset(self):
        self._t = 0
        self.lang_goal_t = random.choice(tokens)
        print('RESET')
        print('LANG GOAL: ', str(self.lang_goal_t))
        return self.client.reset().observation





    def action_shape(self):
        return(8,)

    def get_lang_goal(self):
        return self.client.get_lang_goal()

    def extract_obs(self, obs):
        res_obs = {}

        front_rgb = torch.tensor([copy.deepcopy(obs['image'])], device=self.device)
        front_rgb = front_rgb.permute(0, 3, 1, 2).permute(0, 1, 3, 2).unsqueeze(0)
        res_obs['front_rgb'] = front_rgb

  
        
        front_camera_intrinsics = np.array([[1,0,0],
                                            [0,1,0],
                                            [0,0,1]])

        front_camera_extrinsics = np.array([[-0.56396701,  0.24796995, -0.78768783, 0.0932756],
                                            [ 0.42248543,  0.90620631, -0.01720975, -0.168814],
                                            [ 0.70954019, -0.34249237, -0.61583415, -1.01014076],
                                            [0, 0, 0, 1]], dtype=np.float32)



        res_obs['front_camera_intrinsics'] = torch.tensor([front_camera_intrinsics], device=self.device).unsqueeze(0)
        res_obs['front_camera_extrinsics'] = torch.tensor([front_camera_extrinsics], device=self.device).unsqueeze(0)


        front_point_cloud = obs['point_cloud']
        front_point_cloud = torch.tensor([front_point_cloud], device=self.device)
        front_point_cloud = front_point_cloud.permute(0, 3, 1, 2).permute(0, 1, 3, 2).unsqueeze(0)
        res_obs['front_point_cloud'] = front_point_cloud

        #TODO Не понятно ignore collisions, скорее всего он не работает. У франки он равен 1.0
        res_obs['ignore_collisions'] = torch.tensor([[[0.0]]], device=self.device)


        #if 'descriptions' not in obs.keys():
        #lang_goal_tokens = tokenize(['Put red santa in the box'])[0].numpy()
        lang_goal_tokens = tokenize(self.lang_goal_t)[0].numpy()
        lang_goal_tokens = torch.tensor([lang_goal_tokens], device=self.device).unsqueeze(0)
        res_obs['lang_goal_tokens'] = lang_goal_tokens


        #TODO Надо не забыть про то, что не до конца понятно что такое self._t и верно ли указаны позиции пальцев внутри low_dim_state
        gripper_open = 1 if obs['gripper_pos'] < 0.1 else 0
        time = (1.0 - (self._t/float(self.cfg['rlbench']['episode_length'] - 1))) * 2.0 - 1.0
        low_dim_state = torch.tensor([[[gripper_open, obs['gripper_pos'], obs['gripper_pos'], obs['gripper_is_obj_detected']]]])
        res_obs['low_dim_state'] = low_dim_state

        return res_obs

    def key_msg(self):
        return 'PASS'
        #return self.client.get_key_msg()

    def step(self):
        #TODO key_msg now only in "PASS" mode.

        if self.key_msg()=='PASS':
            print(self.observation['tcp_pose'])
            self.act_result = self.agent.act(self._t, self.extract_obs(self.observation), deterministic=True)
            action = self.act_result.action
            #action[-2] - gripper action
            #print(action[3:7])
            res_act = np.concatenate([action[:7], [(1 - action[-2])*2 - 1], [0]]) #TODO rotation_resolution identify
            print('STEP')
            print(res_act)

            if res_act[2] < 0.043:
                res_act[2] = 0.043


            self.observation = self.client.step(res_act).observation

            terminal = 0

            self._t += 1

            if terminal or self._t >= 4:
                self.observation = self.reset()
            
        elif self.key_msg()=='NEW_LANG_GOAL':
            self.lang_goal = self.get_lang_goal()
            self.observation = self.reset()

        elif self.key_msg()=='RESET':
            self.observation = self.reset()




@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    
    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/saved_seed8_put_in_box_santa_26770/')


    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/put_toy_in_the_box/PERACT_BC/seed6/weights/154000/')
    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/put_toy_in_the_box/PERACT_BC/seed10/weights/133000/')

    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/multi_3/PERACT_BC/seed0/weights/305000/')

    

    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/SelectColor/PERACT_BC/seed1/weights/247000/')
    env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/SelectColor/PERACT_BC/seed3/weights/283000/')
    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/StackCubes/PERACT_BC/seed1/weights/266000/')






    #env = UR5_env(cfg, weights_path='/home/albert/PerActRQC/peract/logs/SanityCheck/PERACT_BC/seed3/weights/97900/')
    res = env.launch()
    while True:
        env.step()

if __name__ == '__main__':
	main()



