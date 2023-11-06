import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from optimal.systems.utils import EarlyStopping

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class GITAI(nn.Module):
    def __init__(self, input_shape, output_shape, device, batch_size=10, hidden_size=128):
        nn.Module.__init__(self)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.base = nn.Sequential(
            init_(nn.Linear(input_shape, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, output_shape))
        )

        self.optimizer = torch.optim.Adam(self.base.parameters(), lr=0.001, eps=1e-5)
        self.earlystopping = EarlyStopping(patience=10, verbose=True)
        self.max_grad_norm = 0.5
        self.batch_size = batch_size

        self.device = device
        self.base.to(device)

    def forward(self, state):
        return self.base(state)
    
    # learn drone dynamics
    def update(self, rollouts):
        train_loss = 0
        eval_loss = 0
        self.base.train()
        train_item_size = 0
        eval_item_size = 0
        gain = 100
        for rollout_batch in rollouts.get(self.batch_size):
            delta_state = gain*(rollout_batch.new_observations[:,:9] - rollout_batch.observations[:,:9])
            # print("state:", rollout_batch.observations[0])
            # print("new_state:", rollout_batch.new_observations[0])
            # print("action:", rollout_batch.actions[0])
            input = torch.cat((rollout_batch.observations[:,3:9], rollout_batch.actions), dim=1)
            pred_delta_state = self.forward(input)
            # print("pred_delta_state:", pred_delta_state[0])
            # print("delta_state:", delta_state[0])
            self.optimizer.zero_grad()
            loss = torch.nn.MSELoss()(delta_state, pred_delta_state)
            nn.utils.clip_grad_norm_(self.parameters(),
                                      self.max_grad_norm)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_item_size += rollout_batch.observations.shape[0]
        self.base.eval()
        with torch.no_grad():
            for rollout_batch in rollouts.get_eval(self.batch_size):
                delta_state = gain*(rollout_batch.new_observations[:,:9] - rollout_batch.observations[:,:9])
                input = torch.cat((rollout_batch.observations[:,3:9], rollout_batch.actions), dim=1)
                pred_delta_state = self.forward(input)
                loss = torch.nn.MSELoss()(delta_state, pred_delta_state)
                eval_loss += loss.item()
                eval_item_size += rollout_batch.observations.shape[0]
        # return train_loss, eval_loss
        return train_loss/train_item_size, eval_loss/eval_item_size
    
    def train(self, rollouts, epochs=10):
        train_loss_array = [0]*epochs
        eval_loss_array = [0]*epochs
        for i in range(epochs):
            train_loss, eval_loss = self.update(rollouts)
            print("Epoch: ", i, "Train loss: ", train_loss, "Eval loss: ", eval_loss)
            train_loss_array[i] = train_loss
            eval_loss_array[i] = eval_loss
            self.earlystopping(eval_loss, self.base)
            if self.earlystopping.early_stop:
                print("Early Stopping!")
                break
        self.save(train_loss_array, eval_loss_array, os.path.join(os.path.dirname(__file__), "../results"))
        
    def save(self, train_loss_array, eval_loss_array, dir):
        torch.save(self.state_dict(), os.path.join(dir, "dynamics_model.pt"))
        # plt.plot(train_loss_array, label="train loss")
        # plt.plot(eval_loss_array, label="eval loss")
        # plt.legend()
        # plt.savefig(os.path.join(dir, "loss.png"))
        #二枚に分ける
        plt.plot(train_loss_array)
        plt.title("train loss")
        plt.savefig(os.path.join(dir, "train_loss.png"))
        plt.plot(eval_loss_array)
        plt.title("eval loss")
        plt.savefig(os.path.join(dir, "eval_loss.png"))
        
        



            

