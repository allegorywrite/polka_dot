import torch
import os
import yaml
import shared_constants

from ppo_wdail.models.polkadot import Polkadot
from gym_pybullet_drones.envs.single_agent_rl.DotAviary import DotAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseDotAviary import ActionType, ObservationType

from ppo_wdail.systems.storage import RolloutStorage
from ppo_wdail.systems.wdail import wdail_train
from ppo_wdail.models.discriminator import Discriminator
from ppo_wdail.systems.dataset import ExpertDataLoader, ExpertDataset

import argparse

# Wasserstein Distance guided Adversarial Imitation Learning (WDAIL)の学習

def train(test_flag=False):
    print("test_flag: ", test_flag)
    print("hello world")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # パラメータの読み込み
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    # 環境の登録
    obs_type = ObservationType.VIS
    act_type = ActionType.VEL5D
    env = DotAviary(params=params,
                    # num_drones=params["env"]["num_drones"],
                    aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                    obs=obs_type,
                    act=act_type,
                    freq=params["env"]["frequency"],
                    goal_position=(1, 1, 1),
                    gui=False,
                    record=False,
                    test_flag=test_flag
                    )
    
    # エージェント(生成モデル)の登録
    obs_space = env.observation_space
    action_space = env.action_space
    print("obs_space: ", obs_space)
    print("action_space: ", action_space)
    num_outputs = action_space.shape[0]
    name = "polkadot"

    agent = Polkadot(
        params=params, 
        obs_space=obs_space, 
        action_space=action_space, 
        num_outputs=num_outputs,
        name=name)
    agent.to(device)

    # 識別モデルの登録
    discriminator = Discriminator(
        params=params,
        device=device
    )

    # データバッファの登録
    rollouts = RolloutStorage(
        num_steps=params["wdail"]["update_steps"],
        num_processes=1,
        obs_shape=obs_space["state"].shape,
        action_space=action_space,
        recurrent_hidden_state_size=agent.recurrent_hidden_state_size
    )

    # データセットの登録
    # gail_train_loader = torch.utils.data.DataLoader(
    #     ExpertDataset(),
    #     batch_size=params["wdail"]["gail_batch_size"],
    #     shuffle=True,
    #     drop_last=True)
    # gail_train_loader = ExpertDataLoader(params=params, data_size=["wdail"]["data_size"], batch_size=params["wdail"]["batch_size"], device=device)
    gail_train_dataset = ExpertDataset(
        params=params, 
        device=device, 
        use_preprocessed_data=True
    )

    # モデルの学習
    wdail_train(
        params=params, 
        env=env, 
        agent=agent, 
        discriminator=discriminator, 
        rollouts=rollouts, 
        gail_train_dataset=gail_train_dataset,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_flag', action='store_true')
    args = parser.parse_args()
    train(test_flag=args.test_flag)