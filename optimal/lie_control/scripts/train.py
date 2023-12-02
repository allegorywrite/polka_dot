"""Script demonstrating the use of `gym_pybullet_drones`' Gym interface.

Class hoverAviary is used as a learning env for the A2C and PPO algorithms.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
The boolean argument --rllib switches between `stable-baselines3` and `ray[rllib]`.
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3` and `ray[rllib]`.
It is not meant as a good/effective learning example.

"""
import time
import argparse
import gym
import numpy as np
# from optimal.lie_control.systems.a2c import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
import torch

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from optimal.lie_control.systems.buffers import DictRolloutBuffer, RolloutBuffer, DataStorage

# from optimal.lie_control.models.gitai import GITAI
from optimal.lie_control.models.SE3FVIN import SE3FVIN
import random
from scipy.spatial.transform import Rotation
torch.set_default_dtype(torch.float32)

DEFAULT_RLLIB = False
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 60
# DEFAULT_SIMULATION_FREQ_HZ = 600
# DEFAULT_CONTROL_FREQ_HZ = 600

def run(
        rllib=DEFAULT_RLLIB,
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        plot=True,
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        debug=False
    ):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz)

    #### Check the environment's spaces ########################
    env = HoverAviary(gui=gui,
                    act=ActionType.DYN,
                    obs=ObservationType.Lie,
                    initial_xyzs=np.array([[0, 0, 1]]),
                    record=record_video,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    )
    print("[INFO] Action space:", env.action_space.shape)
    print("[INFO] Observation space:", env.observation_space.shape)

    batch_size = 1000
    obs_dim = 18
    action_dim = 4

    obs_space_aug = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space_aug = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)

    # dynamics_model = GITAI(
    #     output_shape=env.observation_space.shape[0]-3,
    #     input_shape=env.observation_space.shape[0]-6+env.action_space.shape[0],
    #     device=device, 
    #     batch_size=batch_size)
    dt = 1.0 / DEFAULT_CONTROL_FREQ_HZ
    dynamics_model = SE3FVIN(
        device=device, 
        time_step=dt,
        batch_size=batch_size,
    ).to(device)

    total_timesteps = 3000*env.SIM_FREQ

    data_storage = DataStorage(
            total_timesteps,
            1,
            obs_space_aug,
            action_space_aug,
            device=device,
            n_envs=1,
        )

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    obs = env.reset()
    diff_outputs = np.zeros((1, action_dim))
    raw_obs = np.zeros((1, obs_dim))
    last_obs = raw_obs
    start = time.time()
    for i in range(total_timesteps):
        # action, _states = system_model.predict(obs,deterministic=True)
        if i % 10 == 0:
            if i < total_timesteps*0.2: # warm up
                action = np.array([random.uniform(-1, 1), 0, 0, 0])
            else:
                a = 1
                action = np.array([random.uniform(-1, 1), a*random.uniform(-1, 1), a*random.uniform(-1, 1), a*random.uniform(-1, 1)])
            # action = np.array([0.5, 0, 0, 0.5])
        
        # progress
        if i % 5000 == 0:
            print("Progress: ", int(i/total_timesteps*100), "%")

        obs, reward, done, info = env.step(action)
        actions = np.array([action])
        obses = np.array([obs])
        rewards = np.array([reward])
        dones = np.array([done])
        infos = np.array([info])
        for idx, done in enumerate(dones):
            diff_outputs[idx] = infos[idx]["diff_output"]
            quat = infos[idx]["raw_obs"][3:7]
            R = Rotation.from_quat(quat)
            rotmat = R.as_matrix()
            omega_world = infos[idx]["raw_obs"][13:16]
            omega_body = np.matmul(rotmat.T, omega_world)
            ret = np.hstack([infos[idx]["raw_obs"][0:3], rotmat.flatten(), infos[idx]["raw_obs"][10:13], omega_body]).reshape(18,)
            raw_obs[idx] = ret
        # logger.log(drone=0,
        #            timestamp=i/env.SIM_FREQ,
        #            state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
        #            control=np.zeros(12)
        #            )
        if debug:
            print("action:", action)
            print("obs:", obs)
            print("last_obs:", last_obs)
            print("raw_obs:", raw_obs)
        data_storage.add(
            last_obs,
            raw_obs,
            actions,
            rewards,
        )
        # time.sleep(0.1)
        if gui:
            if i%env.SIM_FREQ == 0:
                env.render()
            sync(i, start, env.TIMESTEP)
            time.sleep(0.1)
        if done:
            obs = env.reset()
            obses = np.array([obs])
        last_obs = raw_obs.copy()
    env.close()

    dynamics_model.update(data_storage, epochs=1000)

    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using hoverAviary')
    parser.add_argument('--rllib',      default=DEFAULT_RLLIB,        type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
