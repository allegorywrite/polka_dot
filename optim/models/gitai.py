import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class GITAI(nn.Module):
    def __init__(self, input_shape, output_shape, device, batch_size, hidden_size=64):
        nn.Module.__init__(self)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.base = nn.Sequential(
            init_(nn.Linear(input_shape, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, output_shape))
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-5)
        self.max_grad_norm = 0.5
        self.batch_size = batch_size

        self.device = device
        self.base.to(device)

    def forward(self, state):
        return self.base(state)
    
    # learn drone dynamics
    def update(self, rollouts):
        update_loss = 0
        for rollout_batch in rollouts.get(self.batch_size):
            # print("state_i:", rollout_batch.observations.shape)
            # print("state_i+1:", rollout_batch.new_observations.shape)
            # print("actions:", rollout_batch.actions.shape)

            delta_state = rollout_batch.new_observations - rollout_batch.observations
            input = torch.cat((rollout_batch.observations, rollout_batch.actions), dim=1)
            pred_delta_state = self.forward(input)
            self.optimizer.zero_grad()
            loss = torch.nn.MSELoss()(delta_state, pred_delta_state)
            nn.utils.clip_grad_norm_(self.parameters(),
                                      self.max_grad_norm)
            loss.backward()
            self.optimizer.step()
            update_loss += loss.item()
        return update_loss
    
    def train(self, rollouts, epochs=10):
        loss_array = [0]*epochs
        for i in range(epochs):
            update_loss = self.update(rollouts)
            print("Epoch: ", i, "Loss: ", update_loss)
            loss_array[i] = update_loss
        self.save(loss_array, os.path.join(os.getcwd(), "../result"))
        
    def save(self, loss, dir):
        torch.save(self.state_dict(), os.path.join(dir, "dynamics_model.pt"))
        plt.plot(loss)
        plt.savefig(os.path.join(dir, "loss.png"))
        
        



            

