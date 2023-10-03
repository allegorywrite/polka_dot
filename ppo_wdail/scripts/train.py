import torch
import os
import yaml
import shared_constants

from swarm.models.polkadot import Polkadot
from gym_pybullet_drones.envs.multi_agent_rl.PolkadotAviary import PolkadotAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

from ppo_wdail.systems.storage import RolloutStorage
from ppo_wdail.systems.wdail import wdail_train
from ppo_wdail.models.discriminator import Discriminator

from ray.rllib.agents import ppo

# Wasserstein Distance guided Adversarial Imitation Learning (WDAIL)の学習

def train():
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
    env = PolkadotAviary(num_drones=params["env"]["num_drones"],
                           aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                           obs=obs_type,
                           act=act_type,
                           freq=params["env"]["frequency"],
                           goal_position=(1, 1, 1),
                           gui=False,
                           record=False
                           )
    
    # エージェント(生成モデル)の登録
    obs_space = env.observation_space[0]
    action_space = env.action_space[0]
    print("obs_space: ", obs_space)
    print("action_space: ", action_space)
    num_outputs = action_space.shape[0]
    model_config = ppo.DEFAULT_CONFIG.copy()
    name = "polkadot"

    agent = Polkadot(obs_space, action_space, num_outputs, model_config, name)

    # 識別モデルの登録
    discriminator = Discriminator()

    # データバッファの登録
    rollouts = RolloutStorage()

    # データセットの登録
    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(),
        batch_size=params["wdail"]["gail_batch_size"],
        shuffle=True,
        drop_last=True)

    # モデルの学習
    wdail_train(params=params, env=env, agent=agent, discriminator=discriminator, rollouts=rollouts, gail_train_loader=gail_train_loader)

if __name__ == "__main__":
    train()