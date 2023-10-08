
import torch.nn as nn
from swarm.models.deepset import DeepSet
from VAE.models.vanilla_vae import VanillaVAE
from gym.spaces import Box, Dict
import torch
import yaml
import os
import numpy as np

from ppo_wdail.models.distributions import Categorical, DiagGaussian, Bernoulli
from ppo_wdail.models.mlp import MLPBase

class Polkadot(nn.Module):
  def __init__(self, obs_space, action_space, num_outputs, name, params=None):
    nn.Module.__init__(self)
    
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

    # MLPの入力空間
    self.mlp_obs_space = Box(low=-1, high=1, shape=(self.own_obs_dim, ))
    # モデルの行動空間
    self.actijon_space = action_space

    self.optimizer = torch.optim.Adam(self.parameters(), lr=params["ppo"]["lr"])

    self.base = MLPBase(self.own_obs_dim, recurrent=False, hidden_size=params["ppo"]["hidden_size"])

    if action_space.__class__.__name__ == "Discrete":
        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)
    elif action_space.__class__.__name__ == "Box":
        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)
    elif action_space.__class__.__name__ == "MultiBinary":
        num_outputs = action_space.shape[0]
        self.dist = Bernoulli(self.base.output_size, num_outputs)
    else:
        raise NotImplementedError
  
  def to(self, device):
    self.device = device
    self.model_neighbors.to(device)
    self.model_obstacle.to(device)
    self.dist.to(device)
    self.base.to(device)
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

  def obs_to_vector(self, input_dict):
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
    return input_ppo

  def forward(self, input_dict, state, seq_lens):
    input_ppo = self.obs_to_vector(input_dict)
    output = self.action_model({"obs" : input_ppo}, state, seq_lens)
    return output

  # def value_function(self):
  #   value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
  #   return torch.reshape(value_out, [-1])

  def get_value(self, obs, rnn_hxs, masks):
    inputs = self.obs_to_vector(obs)
    value, _, _ = self.base(inputs, rnn_hxs, masks)
    return value

  def act(self, obs, rnn_hxs, masks, deterministic=False):
    inputs = self.obs_to_vector(obs)
    value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
    dist = self.dist(actor_features)

    if deterministic:
        action = dist.mode()
    else:
        action = dist.sample()

    action_log_probs = dist.log_probs(action)
    dist_entropy = dist.entropy().mean()

    return value, action, action_log_probs, rnn_hxs
  
  def update(self, rollouts):
    print("Update Generator Model")
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-5)

    value_loss_epoch = 0
    action_loss_epoch = 0
    dist_entropy_epoch = 0

    for e in range(self.ppo_epoch):
        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batch)

        for sample in data_generator:
            obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch, masks_batch,
                actions_batch)

            ratio = torch.exp(action_log_probs -
                              old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()

            if self.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                              value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (return_batch - values).pow(2).mean()

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
              dist_entropy * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                      self.max_grad_norm)
            self.optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

    num_updates = self.ppo_epoch * self.num_mini_batch

    value_loss_epoch /= num_updates
    action_loss_epoch /= num_updates
    dist_entropy_epoch /= num_updates
    total_loss = value_loss_epoch + action_loss_epoch + dist_entropy_epoch

    return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_loss


