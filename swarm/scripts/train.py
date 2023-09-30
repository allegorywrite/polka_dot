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
import open3d as o3d
import pandas as pd
import quaternion
from swarm.system.sim import SimulationManager

from gym_pybullet_drones.envs.multi_agent_rl.PolkadotAviary import PolkadotAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
import matplotlib.pyplot as plt

import pickle

def make_dataset(params, device, use_preprocessed_data = False, idx=None):
    # キャッシュが存在する場合はそれを読み込む
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx))):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)), 'rb') as f:
            dataset = pickle.load(f)
        print("dataset loaded from ", os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)))
        return dataset
    loader = CustomDataset()
    train_dataset_dict, test_dataset_dict = loader.load_data(load=use_preprocessed_data, save=True, visualize=False, id=idx)
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
            goal = np.array(state[1]).flatten()
            velocity = np.array(state[2]).flatten()
            neighbor = np.array(state[3]).flatten()
            depth = np.array(state[4][0]).flatten()
            combined = np.concatenate((goal, velocity, depth, neighbor))
            batch_x.append(combined)
            batch_y.append(action)
        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)
        # バッチサイズに分割してtorchに変換
        for i in range(0, len(batch_x), params["ppo"]["batch_size"]):
            batch_x_torch = torch.from_numpy(batch_x[i:i+params["ppo"]["batch_size"]])
            batch_y_torch = torch.from_numpy(batch_y[i:i+params["ppo"]["batch_size"]])
            dataset.append([batch_x_torch, batch_y_torch])
        # batch_x_torchのサイズは, 3(goal position) + 3(velocity) + 6*neighbor_num + hidden state
    print("dataset size: ", len(dataset), "x", params["ppo"]["batch_size"])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)), 'wb') as f:
        pickle.dump(dataset, f)
    return dataset

def step(state, action):
    # action:[v_x, v_y, v_z, |v|, w_z]

    SPEED_LIMIT = 1000.0

    time = np.array([0])
    p_new = state[0] + SPEED_LIMIT*np.abs(action[3])*action[0:3]
    v_new = SPEED_LIMIT*np.abs(action[3])*action[0:3]
    # q_new = np.array([state[2].w, state[2].x, state[2].y, state[2].z])
    quaternion_iw = quaternion.from_euler_angles([0, 0, action[4]]) * state[2]
    q_new = np.array([quaternion_iw.w, quaternion_iw.x, quaternion_iw.y, quaternion_iw.z])
    w_new = np.array([0, 0, action[4]])
    
    new_state = np.concatenate([time, p_new, q_new, v_new, w_new])
    return new_state

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_preprocessed_data", action="store_true")
    parser.add_argument("--use_pretrained_model", action="store_true")
    args = parser.parse_args()

    # キャッシュディレクトリの作成
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/')):
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/'))
    # キャッシュの初期化
    for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/')):
        os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/') + file)

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
                           gui=False,
                           record=False
                           )
    
    obs_space = temp_env.observation_space[0]
    action_space = temp_env.action_space[0]
    # print("obs_space: ", obs_space)
    # print("action_space: ", action_space)
    num_outputs = action_space.shape[0]
    model_config = ppo.DEFAULT_CONFIG.copy()
    name = "polkadot"
    # print("num_outputs: ", num_outputs)

    model = Polkadot(obs_space, action_space, num_outputs, model_config, name)
    model.to(device)

    if args.use_pretrained_model:
        model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output/model.pth')
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print("model loaded from ", model_load_path)
    else:
        # データセットの作成
        # dataset = make_dataset(params, device, args.use_preprocessed_data)

        optimizer = Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["ppo"]["scheduler_gamma"])
        epochs = 200

        model.train()
        epoch_losses = []
        print("start training")

        for epoch in range(epochs):
            epoch_loss = 0
            # TODO
            file_batch_num = 110
            for idx in range(file_batch_num):
                batch_dataset = make_dataset(params, device, args.use_preprocessed_data, idx)
                if len(batch_dataset) == 0:
                    break
                for idx, (batch_x, batch_y) in enumerate(batch_dataset):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    output, state = model.forward_pretrain(batch_x)
                    loss = torch.nn.MSELoss()(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                del batch_dataset
            epoch_losses.append(epoch_loss)
            print("epoch: ", epoch, " loss: ", epoch_loss)
            scheduler.step()


        # for epoch in range(epochs):
        #     epoch_loss = 0
        #     for idx, (batch_x, batch_y) in enumerate(dataset):
        #         batch_x = batch_x.to(device)
        #         batch_y = batch_y.to(device)
        #         optimizer.zero_grad()
        #         output, state = model.forward_pretrain(batch_x)
        #         loss = torch.nn.MSELoss()(output, batch_y)
        #         loss.backward()
        #         optimizer.step()
        #         epoch_loss += loss.item()
        #     epoch_losses.append(epoch_loss)  # Append the epoch_loss to the end of the list
        #     print("epoch: ", epoch, " loss: ", epoch_loss)
        #     scheduler.step()
            
        # Plotting
        plt.plot(epoch_losses)
        plt.title('Epoch Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        model.eval()
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output/model.pth')
        torch.save(model.state_dict(), model_save_path)
        print("model saved at ", model_save_path)

    #### Test the model ####
    
    # シミュレーションのセットアップ
    sim_manager = SimulationManager()
    # ファイルを読み込み
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    map_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/map/{}.yaml".format(params["env"]["map_name"]))
    vision_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/vision/{}.pcd".format(params["env"]["map_name"]))
    replay_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/replay/{}.csv".format(params["env"]["map_name"]))
    map_data = yaml.load(open(map_file_path), Loader=yaml.FullLoader)
    global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
    df = pd.read_csv(replay_file_path, header=None)
    replay_data = df.to_numpy()
    # 初期状態
    simplay_data = np.empty((0, sim_manager.dim_per_drone * sim_manager.drones_num))
    # simplay_data = replay_data[0]
    simplay_data = np.vstack([simplay_data, replay_data[0]])
    # シミュレーション
    sim_manager.create_field(global_map_world_pc)
    sim_count = 10
    for sim_itr in range(sim_count):
        snapshot_data = []
        for agent_id in range(sim_manager.drones_num):
            # get observation
            p_i_world, q_i_world, v_i_world, w_i_world, goal_i = sim_manager.getSOA_of_world(simplay_data, map_data, agent_id=agent_id, t=sim_itr, sim=True)
            state = [p_i_world, v_i_world, q_i_world, w_i_world]
            neighbor_state_local_array, v_i_local, goal_local = sim_manager.transform_to_local(simplay_data, sim_itr, agent_id, p_i_world, q_i_world, v_i_world, goal_i)
            normalized_neighbor_state_local_array, normalized_v_i_local, normalized_goal_local = sim_manager.clip_and_normlize(neighbor_state_local_array, v_i_local, goal_local)
            depth_data = sim_manager.get_local_observation(p_i_world, q_i_world)
            # prediction step

            # obs = {
            #     "state": np.concatenate((np.array([len(normalized_neighbor_state_local_array)/6]), normalized_goal_local, normalized_v_i_local)),
            #     "neighbors": normalized_neighbor_state_local_array,
            #     "depth": depth_data,
            # }
            # action, _ = model.forward(obs, None, None)
            # # dynamical step
            # print("action: ", action.cpu().detach().numpy()[0])
            # new_state = step(state, action.cpu().detach().numpy()[0])

            action = np.array([0.5, 0.3, 0.2, 0.001, -0.01])
            new_state = step(state, action)

            snapshot_data.extend(new_state)
        simplay_data = np.vstack([simplay_data, snapshot_data])
    sim_manager.visualize_open3d_world(replay_data=simplay_data, t=sim_itr, global_map_world_pc=global_map_world_pc, p_i_world=p_i_world, q_i_world=q_i_world)
    sim_manager.destroy_field(global_map_world_pc)

    #### Show (and record a video of) the model's performance ####
    # obs = temp_env.reset()
    # logger = Logger(logging_freq_hz=int(temp_env.SIM_FREQ/temp_env.AGGR_PHY_STEPS),
    #                 num_drones=temp_env.NUM_DRONES
    #                 )
    # start = time.time()
    # log_time = 5
    # for i in range(5*int(temp_env.SIM_FREQ/temp_env.AGGR_PHY_STEPS)):
    #     # actionを計算
    #     action_dict = {}
    #     for agent_id in range(temp_env.NUM_DRONES):
    #         action, state = model.forward(obs[agent_id], None, None)
    #         action_dict[agent_id] = action.cpu().detach().numpy()[0]
    #     obs, reward, done, info = temp_env.step(action_dict)
    #     temp_env.render()
    #     sync(np.floor(i*temp_env.AGGR_PHY_STEPS), start, temp_env.TIMESTEP)
    # temp_env.close()
    # logger.save_as_csv("polkadot")
    # logger.plot()