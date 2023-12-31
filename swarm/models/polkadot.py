
import torch.nn as nn
from swarm.models.deepset import DeepSet
from VAE.models.vanilla_vae import VanillaVAE
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from gym.spaces import Box, Dict
import torch
import yaml
import os

class Polkadot(TorchModelV2, nn.Module):
  def __init__(self, obs_space, action_space, num_outputs, model_config, name, device):
    TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
    nn.Module.__init__(self)

    self.device = device

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
      params = yaml.load(f, Loader=yaml.SafeLoader)
    
    # 状態
    self.state_dim = 6
    # DeepSetの出力次元
    self.deepset_latent_dim = params["deepset"]["latent_dim"]
    # VAEの出力次元
    self.vae_latent_dim = params["vae"]["latent_dim"]
    # PPOの入力次元
    self.own_obs_dim = self.state_dim + self.vae_latent_dim + self.deepset_latent_dim

    self.model_neighbors = DeepSet(self.state_dim, self.deepset_latent_dim)
    self.model_obstacle = VanillaVAE(in_channels=params["vae"]["in_channels"], latent_dim=self.vae_latent_dim)
    model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
    self.model_obstacle.load_state_dict(torch.load(model_load_path, map_location=self.device))
    self.model_obstacle.eval()
    self.action_model = FullyConnectedNetwork(
      Box(low=-1, high=1, shape=(self.own_obs_dim, )), 
      action_space,
      num_outputs,
      model_config,
      name + "_action"
      )
    self.value_model = FullyConnectedNetwork(
      Box(low=-1, high=1, shape=(self.own_obs_dim, )), 
      action_space,
      1, 
      model_config, 
      name + "_vf"
      )
    self._model_in = None
    self.optimizer = torch.optim.Adam(self.parameters(), lr=params["ppo"]["lr"])
  
  def to(self, device):
    self.device = device
    self.model_neighbors.to(device)
    self.model_obstacle.to(device)
    self.action_model.to(device)
    self.value_model.to(device)
    return super().to(device)
  
  # input = [g ∈ R^3, v ∈ R^3, depth ∈ R^hidden_dim, neighbor_0 ~ neighbor_n ∈ R^6]
  def forward_pretrain(self, input):
    input_neighbors = input[:, 6+self.vae_latent_dim:]
    output_neighbors = self.model_neighbors(input_neighbors)
    input_ppo = torch.cat([input[:, :6+self.vae_latent_dim], output_neighbors], dim=1)
    output_ppo = self.action_model({"obs" : input_ppo}, None, None)
    return output_ppo

  """
  Parameters
  ----------
  state : [|N|, gx, gy, gz, vx, vy, vz]
  neighbors : [x0, y0, z0, vx0, vy0, vz0, ..., xN, yN, zN, vxN, vyN, vzN]
  depth : [depth]

  PPO Input
  ---------
  obs : [gx, gy, gz, vx, vy, vz, vae_output, deepset_output]

  Returns
  -------
  action : [vx, vy, vz, |v|, wz]
  """
  def forward(self, input_dict, state, seq_lens):
    # print("state:", input_dict["state"])
    # print("depth:", input_dict["depth"])
    # print("neighbors:", input_dict["neighbors"])
    neighbors_num = input_dict["state"][0]
    # Deep Set
    input_neighbors = input_dict["neighbors"][0:int(neighbors_num*6)]
    # Tesorに変換
    input_neighbors = torch.from_numpy(input_neighbors).unsqueeze(0).to(self.device)
    deepset_output = self.model_neighbors(input_neighbors)
    # VAE
    input_obs = torch.from_numpy(input_dict["depth"]).unsqueeze(0).unsqueeze(0).to(self.device)
    mu, log_var = self.model_obstacle.encode(input_obs)
    vae_output = self.model_obstacle.reparameterize(mu, log_var)

    input_state = torch.from_numpy(input_dict["state"][1:]).unsqueeze(0).to(self.device)
    input_ppo = torch.cat([input_state, vae_output, deepset_output], dim=1)
    # TODO
    # self._model_in = [input_dict["obs_flat"], state, seq_lens]
    self._model_in = [input_ppo, state, seq_lens]
    output = self.action_model({"obs" : input_ppo}, state, seq_lens)
    return output

  def value_function(self):
    value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
    return torch.reshape(value_out, [-1])

  
