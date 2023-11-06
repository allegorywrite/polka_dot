import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from swarm.models.deepset import DeepSet
from VAE.models.vanilla_vae import VanillaVAE

from baselines.common.running_mean_std import RunningMeanStd

class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels

class Discriminator(nn.Module):
    def __init__(self, params, device):
        super(Discriminator, self).__init__()
        self.params = params
        self.cliprew_down = params["discriminator"]["cliprew_down"]
        self.cliprew_up = params["discriminator"]["cliprew_up"]
        self.device = device
        self.reward_type = params["discriminator"]["reward_type"]
        self.update_rms = params["discriminator"]["update_rms"]

        # 状態
        self.state_dim = 6
        # DeepSetの出力次元
        self.deepset_latent_dim = params["deepset"]["latent_dim"]
        # VAEの出力次元
        self.vae_latent_dim = params["vae"]["latent_dim"]
        # 状態次元
        self.own_obs_dim = self.state_dim + self.vae_latent_dim + self.deepset_latent_dim
        # 行動次元
        self.action_dim = params["env"]["action_dim"]

        self.model_neighbors = DeepSet(self.state_dim, self.deepset_latent_dim).to(device)

        input_dim = self.own_obs_dim + self.action_dim
        hidden_dim = params["discriminator"]["hidden_dim"]

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, obs, action):
        if type(obs) == dict:
            print("Error in Discriminator.forward() type(obs) = %s" % type(obs))
            raise NotImplementedError
        elif type(obs) == torch.Tensor:
            input_neighbors = obs[:, self.state_dim+self.vae_latent_dim:]
            output_neighbors = self.model_neighbors(input_neighbors)
            state = torch.cat([obs[:, :self.state_dim+self.vae_latent_dim], output_neighbors], dim=1)
        else:
            print("Error in Discriminator.forward() type(obs) = %s" % type(obs))
            raise NotImplementedError
        
        return self.trunk(torch.cat([state.to(torch.float32), action.to(torch.float32)], dim=1))
    
    def obs_to_state(self, obs):
        if type(obs) == dict:
            print("Error in Discriminator.forward() type(obs) = %s" % type(obs))
            raise NotImplementedError
        elif type(obs) == torch.Tensor:
            input_neighbors = obs[:, self.state_dim+self.vae_latent_dim:]
            output_neighbors = self.model_neighbors(input_neighbors)
            state = torch.cat([obs[:, :self.state_dim+self.vae_latent_dim], output_neighbors], dim=1)
        else:
            print("Error in Discriminator.forward() type(obs) = %s" % type(obs))
            raise NotImplementedError
        return state

    def compute_grad_pen(self,
                         expert_obs,
                         expert_action,
                         policy_obs,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_obs.size(0), 1)
        expert_data = torch.cat([expert_obs, expert_action], dim=1)
        policy_data = torch.cat([policy_obs, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        # disc = self.trunk(mixup_data)
        disc = self.forward(mixup_data[:, :-self.action_dim], mixup_data[:, -self.action_dim:])
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            # policy_d = self.trunk(
            #     torch.cat([policy_state, policy_action], dim=1))
            policy_d = self.forward(policy_state, policy_action)

            expert_state, expert_action = expert_batch
            # expert_state = obsfilt(expert_state.numpy(), update=False)
            # expert_state = torch.FloatTensor(expert_state).to(self.device)
            # expert_action = expert_action.to(self.device)
            # expert_d = self.trunk(
            #     torch.cat([expert_state, expert_action], dim=1))
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.forward(expert_state, expert_action)
            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))

            # expert_loss = torch.mean(expert_d).to(self.device)
            # policy_loss = torch.mean(policy_d).to(self.device)

            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            # gail_loss = expert_loss + policy_loss
            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            # loss += (gail_loss + grad_pen).item()
            loss += (-wd + grad_pen).item()
            g_loss += (wd).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (-wd + grad_pen).backward()
            self.optimizer.step()

        return g_loss/n, gp/n, 0.0, loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            # d = self.trunk(torch.cat([state, action], dim=1))
            d = self.forward(state, action)
            if self.reward_type == 0:
                s = torch.exp(d)
                reward = s
            elif self.reward_type == 1:
                s = torch.sigmoid(d)
                reward = - (1 - s).log()
            elif self.reward_type == 2:
                s = torch.sigmoid(d)
                reward = s
            elif self.reward_type == 3:
                s = torch.sigmoid(d)
                reward = s.exp()
            elif self.reward_type == 4:
                reward = d
            elif self.reward_type == 5:
                s = torch.sigmoid(d)
                reward = s.log() - (1 - s).log()

            # s = torch.exp(d)
            # # reward = s.log() - (1 - s).log()
            # s = torch.sigmoid(d)
            # reward = s
            # # reward = d
            if self.returns is None:
                self.returns = reward.clone()

            if self.update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward


            # ttt = torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return torch.clamp(reward / np.sqrt(self.ret_rms.var[0] + 1e-8), self.cliprew_down, self.cliprew_up)
            # return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            # return torch.clamp(reward,self.cliprew_down, self.cliprew_up)
            # return reward