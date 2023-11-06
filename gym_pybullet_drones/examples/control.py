"""Script demonstrating the use of `gym_pybullet_drones`' Gymnasium interface.

Class HoverAviary is used as a learning env for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning libraries `stable-baselines3`.

"""
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 240

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
    env = HoverAviary(gui=gui,
                    act=ActionType.DYN,
                    initial_xyzs=np.array([[0, 0, 2]]),
                    record=record_video,
                    freq=simulation_freq_hz,
                    aggregate_phy_steps=AGGR_PHY_STEPS,
                    )
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    #### Train the model #######################################
    model = PPO("MlpPolicy",
                env,
                verbose=1
                )
    
    # model = PPO.load("ppo_hover", env=env)
    # model.learn(total_timesteps=1000000) # Typically not enough
    # model.save("ppo_control")
    # model = PPO.load("ppo_hover", env=env)

    #### Show (and record a video of) the model's performance ##
    # env = HoverAviary(gui=gui,
    #                   act=ActionType.DYN,
    #                   record=record_video
    #                  )
    # logger = Logger(logging_freq_hz=1,
    #                 num_drones=1,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 )
    # obs, info = env.reset()
    # start = time.time()
    # for i in range(3*env.CTRL_FREQ):
    #     action, _states = model.predict(obs,
    #                                     deterministic=True
    #                                     )
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     logger.log(drone=0,
    #                timestamp=i/env.CTRL_FREQ,
    #                state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
    #                control=np.zeros(12)
    #                )
    #     env.render()
    #     print(terminated)
    #     sync(i, start, env.CTRL_TIMESTEP)
    #     if terminated:
    #         obs = env.reset()
    # env.close()

    #### Show (and record a video of) the model's performance ##
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
        time.sleep(0.01)
        logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                   control=np.zeros(12)
                   )
        if i%env.SIM_FREQ == 0:
            env.render()
        sync(i, start, env.TIMESTEP)
        if done:
            obs = env.reset()
    env.close()

    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using HoverAviary')
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))