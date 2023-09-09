import sys
sys.path.append("../../")

import torch
from torch.optim import Adam
from swarm.models.polkadot import Polkadot
from ray.tune import register_env
from ray.rllib.agents import ppo
from swarm.models.dataset import CustomDataset
from VAE.models.vanilla_vae import VanillaVAE
import torch
import os
import numpy as np
import argparse
import yaml
import shared_constants
import time

from gym_pybullet_drones.envs.multi_agent_rl.PolkadotAviary import PolkadotAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

def make_dataset(params, device, use_preprocessed_data = False):
    loader = CustomDataset(False)
    train_dataset_dict, test_dataset_dict = loader.load_data(use_preprocessed_data)
    model = VanillaVAE(in_channels = params["vae"]["in_channels"], latent_dim=params["vae"]["latent_dim"])
    model.to(device)
    model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()
    dataset = []
    # 周辺UAV数でデータを分割
    for key in train_dataset_dict.keys():
        print("agent_num: ", key, " train_dataset: ", len(train_dataset_dict[key]))
        train_dataset = train_dataset_dict[key]
        batch_x = []
        batch_y = []
        # デプス画像を潜在変数に変換
        for train_data in train_dataset:
            depth = train_data[4]
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
            mu, log_var = model.encode(depth)
            z = model.reparameterize(mu, log_var)
            train_data[4] = z.cpu().detach().numpy()
            state = np.array(train_data)[0:-1]
            action = np.array(train_data)[-1]
            # print("velocity: ", state[2])
            # print("neighbor:", state[0], "goal: ",len(state[1]), " velocity: ",len(state[2]), " neighbor: ",state[3], " depth: ",len(state[4][0]))
            goal = np.array(state[1]).flatten()
            velocity = np.array(state[2]).flatten()
            neighbor = np.array(state[3]).flatten()
            depth = np.array(state[4][0]).flatten()
            combined = np.concatenate((goal, velocity, depth, neighbor))
            # print("combined: ",combined.shape)
            # print("action size: ",len(action))
            batch_x.append(combined)
            batch_y.append(action)
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)
        # バッチサイズに分割してtorchに変換
        for i in range(0, len(batch_x), params["ppo"]["batch_size"]):
            batch_x_torch = torch.from_numpy(batch_x[i:i+params["ppo"]["batch_size"]])
            batch_y_torch = torch.from_numpy(batch_y[i:i+params["ppo"]["batch_size"]])
            # print("batch_x_torch: ", batch_x_torch.shape)
            # print("batch_y_torch: ", batch_y_torch.shape)
            dataset.append([batch_x_torch, batch_y_torch])
        # batch_x_torchのサイズは, 3(goal position) + 3(velocity) + 6*neighbor_num + hidden state
    print("dataset size: ", len(dataset), "x", params["ppo"]["batch_size"])
    return dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_preprocessed_data", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # パラメータの読み込み
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    # 環境の登録
    env_name = "this-aviary-v0"

    # TODO: 行動空間と観測空間の実装
    obs_type = ObservationType.VIS
    act_type = ActionType.VEL5D
    register_env(env_name, lambda _: PolkadotAviary(num_drones=params["env"]["num_drones"],
                                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                obs=obs_type,
                                                act=act_type,
                                                freq=params["env"]["frequency"]
                                                )
    )
    # パラメータ取得用のenv
    temp_env = PolkadotAviary(num_drones=params["env"]["num_drones"],
                           aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                           obs=obs_type,
                           act=act_type,
                           freq=params["env"]["frequency"],
                           goal_position=(1, 1, 1),
                           gui=True,
                           record=False
                           )
    
    obs_space = temp_env.observation_space[0]
    action_space = temp_env.action_space[0]
    print("obs_space: ", obs_space)
    print("action_space: ", action_space)
    num_outputs = action_space.shape[0]
    model_config = ppo.DEFAULT_CONFIG.copy()
    name = "polkadot"
    print("num_outputs: ", num_outputs)

    model = Polkadot(obs_space, action_space, num_outputs, model_config, name)
    model.to(device)

    # データセットの作成
    dataset = make_dataset(params, device, args.use_preprocessed_data)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["ppo"]["scheduler_gamma"])
    epochs = 10

    model.train()
    print("start training")
    for epoch in range(epochs):
        epoch_loss = 0
        for idx, (batch_x, batch_y) in enumerate(dataset):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output, state = model.forward_pretrain(batch_x)
            loss = torch.nn.MSELoss()(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch: ", epoch, " loss: ", epoch_loss)
        scheduler.step()
    model.eval()
    model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output/model.pth')
    torch.save(model.state_dict(), model_save_path)
    print("model saved at ", model_save_path)

    #### Show (and record a video of) the model's performance ####
    obs = temp_env.reset()
    logger = Logger(logging_freq_hz=int(temp_env.SIM_FREQ/temp_env.AGGR_PHY_STEPS),
                    num_drones=temp_env.NUM_DRONES
                    )
    start = time.time()
    log_time = 5
    for i in range(5*int(temp_env.SIM_FREQ/temp_env.AGGR_PHY_STEPS)):
        # actionを計算
        action_dict = {}
        for agent_id in range(temp_env.NUM_DRONES):
            action = model.forward(obs[agent_id], None, None)
            action_dict[agent_id] = action.detach().numpy()
        obs, reward, done, info = temp_env.step(action)
        temp_env.render()
        sync(np.floor(i*temp_env.AGGR_PHY_STEPS), start, temp_env.TIMESTEP)
    temp_env.close()
    logger.save_as_csv("polkadot")
    logger.plot()

        
