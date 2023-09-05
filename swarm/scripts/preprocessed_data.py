from swarm.models.dataset import CustomDataset
from VAE.models.vanilla_vae import VanillaVAE
import torch
import os
import numpy as np
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_preprocessed_data", action="store_true")
  args = parser.parse_args()

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  
  in_channels = 1
  latent_dim = 200 # 潜在変数の次元数
  batch_size = 32
  loader = []

  # データセットの読み込み
  dataset = CustomDataset(False)
  # train_dataset_dict, test_dataset_dictは
  # array of [neighbor_num ∈ N, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action ∈ R^3] の辞書データ
  train_dataset_dict, test_dataset_dict = dataset.load_data(args.use_preprocessed_data)
  # エンコーダーの読み込み
  model = VanillaVAE(in_channels = in_channels, latent_dim=latent_dim)
  model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
  model.load_state_dict(torch.load(model_load_path, map_location=device))
  model.to(device)
  model.eval()
  for key in train_dataset_dict.keys():
    print("agent_num: ", key, " train_dataset: ", len(train_dataset_dict[key]))
    train_dataset = train_dataset_dict[key]
    batch_x = []
    batch_y = []
    # デプス画像を潜在変数に変換
    for train_data in train_dataset:
      depth = train_data[4]
      # depth = depth.unsqueeze(0).to(device)
      depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)
      mu, log_var = model.encode(depth)
      z = model.reparameterize(mu, log_var)
      train_data[4] = z.cpu().detach().numpy()
      state = np.array(train_data)[0:-1]
      action = np.array(train_data)[-1]
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
    for i in range(0, len(batch_x), batch_size):
      batch_x_torch = torch.from_numpy(batch_x[i:i+batch_size])
      batch_y_torch = torch.from_numpy(batch_y[i:i+batch_size])
      print("batch_x_torch: ", batch_x_torch.shape)
      print("batch_y_torch: ", batch_y_torch.shape)
      loader.append([batch_x_torch, batch_y_torch])
    # batch_x_torchのサイズは, 3(goal position) + 3(velocity) + 6*neighbor_num + hidden state




      

  