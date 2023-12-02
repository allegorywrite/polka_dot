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

    total_sequences = 200*env.SIM_FREQ
    seq_step = 10
    total_timesteps = total_sequences*seq_step

    data_storage = DataStorage(
            total_sequences,
            1,
            obs_space_aug,
            action_space_aug,
            device=device,
            n_envs=seq_step,
        )

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    
    obs, info = env.reset()
    start = time.time()
    raw_obs = np.zeros((seq_step, obs_dim))
    obses = np.zeros((seq_step, *obs.shape))
    actions = np.zeros((seq_step, action_dim))
    diff_outputs = np.zeros((seq_step, action_dim))
    dones = np.zeros((seq_step))
    rewards = np.zeros((seq_step))
    last_obs = np.zeros((seq_step, obs_dim))

    quat = info["raw_obs"][3:7]
    R = Rotation.from_quat(quat)
    rotmat = R.as_matrix()
    omega_world = info["raw_obs"][13:16]
    omega_body = np.matmul(rotmat.T, omega_world)
    last_obs[0] = np.hstack([info["raw_obs"][0:3], rotmat.flatten(), info["raw_obs"][10:13], omega_body]).reshape(18,)
    itr = 0

    # for i in range(total_sequences):
    while itr < total_sequences:
        if itr % 1000 == 0:
            print("Progress: ", int(itr/total_sequences*100), "%")
        valid_seq = True

        if itr < total_sequences*0.6: # warm up
            action = np.array([random.uniform(-1, 1), 0, 0, 0])
        else:
            a = 1
            action = np.array([random.uniform(-1, 1), a*random.uniform(-1, 1), a*random.uniform(-1, 1), a*random.uniform(-1, 1)])
        
        for j in range(seq_step):
            
            obs, reward, done, info = env.step(action)
            actions[j] = action
            obses[j] = obs
            rewards[j] = reward
            dones[j] = done

            diff_outputs[j] = info["diff_output"]
            quat = info["raw_obs"][3:7]
            R = Rotation.from_quat(quat)
            rotmat = R.as_matrix()
            omega_world = info["raw_obs"][13:16]
            omega_body = np.matmul(rotmat.T, omega_world)
            ret = np.hstack([info["raw_obs"][0:3], rotmat.flatten(), info["raw_obs"][10:13], omega_body]).reshape(18,)
            raw_obs[j] = ret

            if debug:
                print("action:", action)
                print("obs:", obs)
                print("last_obs:", last_obs)
                print("raw_obs:", raw_obs)
            if j < seq_step-1:
                last_obs[j+1] = raw_obs[j].copy()

            if gui:
                if itr%env.SIM_FREQ == 0:
                    env.render()
                sync(itr, start, env.TIMESTEP)
                time.sleep(0.1)
            if done:
                valid_seq = False
                break

        if valid_seq:
            data_storage.add(
                last_obs,
                raw_obs,
                actions,
                rewards,
            )
            last_obs[0] = raw_obs[-1].copy()
            itr += 1
        else:
            obs, info = env.reset()
            last_obs[0] = np.hstack([info["raw_obs"][0:3], rotmat.flatten(), info["raw_obs"][10:13], omega_body]).reshape(18,)
        
    env.close()
    dynamics_model.update(data_storage, epochs=1000, seq_len=seq_step)

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
