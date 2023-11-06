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
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 60

def run(
        output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, plot=True, 
        colab=DEFAULT_COLAB, 
        record_video=DEFAULT_RECORD_VIDEO,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    ):

    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz)

    #### Check the environment's spaces ########################
    # env = gym.make("hover-aviary-v0")
    # env = HoverAviary(gui=gui,
    #                 record=record_video,
    #                 act=ActionType.DYN,
    #                 )
    env = HoverAviary(gui=gui,
                    act=ActionType.VEL,
                    initial_xyzs=np.array([[0, 0, 0.5]]),
                    record=record_video,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    )
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    #### Train the model #######################################
    # model = A2C(MlpPolicy,
    #             env,
    #             verbose=1
    #             )
    # model.learn(total_timesteps=500000)
    # model.save("a2c_hover")
    model = A2C.load("a2c_hover", env=env)

    #### Show (and record a video of) the model's performance ##
    # env = HoverAviary(gui=gui,
    #                     record=record_video
    #                     )
    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs = env.reset()
    start = time.time()
    for i in range(10*env.SIM_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, done, info = env.step(action)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
            print(done)
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()

    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using hoverAviary')
    # parser.add_argument('--rllib',      default=DEFAULT_RLLIB,        type=str2bool,       help='Whether to use RLlib PPO in place of stable-baselines A2C (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))