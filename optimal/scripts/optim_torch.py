import numpy as np
import torch
from optimal.models.gitai import GITAI
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import matplotlib.pyplot as plt

class Optimizer:

    def __init__(self, initial_x, u_seq, x_target, model=None, model_based=False, device=torch.device('cpu')):
        self.x_seq = torch.zeros((len(u_seq)+1, initial_x.shape[0])).to(device)
        self.x_seq[0] = torch.from_numpy(initial_x).float().to(device)
        # self.u_seq = np.array(u_seq)
        self.u_seq = torch.from_numpy(u_seq).float().to(device)
        # print("u_seq_torch:", self.u_seq_torch.shape)
        self.model = model
        self.device = device
        self.x_target = [None for _ in range(len(x_target))]
        for i in range(len(x_target)):
            if x_target[i] is not None:
                self.x_target[i] = torch.from_numpy(np.array(x_target[i])).float().to(device)
        self.model_based = model_based
        self.seq_len = u_seq.shape[0]
        self.compute_x_seq(self.u_seq)

    # (roll pitch yaw)がそれぞれ(x, y, z)の微分値であるようなモデルを考える
    def model_based_model(self, x, u): # tensor -> tensor
        delta = torch.zeros(x.shape[0]).to(self.device)
        delta[0] = u[1]
        delta[1] = u[2]
        delta[2] = u[3]
        return delta

    def model_based_grad_f(self, x, u):
        du = torch.zeros((x.shape[0], u.shape[0])).to(self.device)
        dx = torch.zeros((x.shape[0], x.shape[0])).to(self.device)
        du[0,1] = 1
        du[1,2] = 1
        du[2,3] = 1
        return dx, du

    def compute_x_seq(self, u_seq):

        for i in range(self.seq_len):
            if self.model_based:
                self.x_seq[i+1] = self.x_seq[i] + self.model_based_model(self.x_seq[i], u_seq[i])
            else:
                input = torch.cat((self.x_seq[i], u_seq[i]), 0)
                self.x_seq[i+1] = self.model(input).detach()

        # u_seq_torch = torch.from_numpy(u_seq).float().to(self.device)
        # x_seq_torch = torch.from_numpy(self.x_seq).float().to(self.device)
        # if self.model_based:
        #     self.x_seq[1:] = self.x_seq[:-1] + self.model_based_model(x_seq_torch[:-1], u_seq_torch[:-1])
        # else:
        #     input = torch.cat((x_seq_torch[:-1], u_seq_torch[:-1]), dim=1)
        #     self.x_seq[1:] = self.model(input).cpu().numpy()

    def grad_f(self, x, u):
        if self.model_based:
            return self.model_based_grad_f(x, u)
        # x = torch.from_numpy(x).float().to(self.device)
        # u = torch.from_numpy(u).float().to(self.device)
        x.requires_grad_(True)
        u.requires_grad_(True)
        input = torch.cat((x, u), 0)
        delta = self.model(input)
        x_grad_mat = torch.zeros((x.shape[0], x.shape[0])).to(self.device)
        u_grad_mat = torch.zeros((x.shape[0], u.shape[0])).to(self.device)
        for i in range(delta.shape[0]):
            x_grad = torch.autograd.grad(delta[i], x, create_graph=True)[0]
            x_grad_mat[i] = x_grad.detach()
            u_grad = torch.autograd.grad(delta[i], u, create_graph=True)[0]
            u_grad_mat[i] = u_grad.detach()
        
        return x_grad_mat, u_grad_mat

    def grad_xu(self, i, j):
        grad = torch.zeros((self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        if j < i - 1:
            grad = torch.mm(self.grad_xx(i, i-1),self.grad_xu(i-1, j)).to(self.device)
        elif j == i - 1:
            grad = self.grad_f(self.x_seq[i-1], self.u_seq[i-1])[1]
        return grad

    def grad_xx(self, i, j):
        grad = torch.zeros((self.x_seq.shape[1], self.x_seq.shape[1])).to(self.device)
        if j < i -1:
            grad = torch.dot(self.grad_xx(i, i-1),self.grad_xx(i-1, j)).to(self.device)
        elif j == i - 1:
            grad = torch.eye(self.x_seq.shape[1]).to(self.device) + self.grad_f(self.x_seq[i-1], self.u_seq[i-1])[0]
        elif j == i:
            grad = torch.eye(self.x_seq.shape[1]).to(self.device)
        return grad

    def partial_lx(self, i):
        # 目的関数に依存
        omega_target = 1
        grad = torch.zeros(self.x_seq.shape[1]).to(self.device)
        if type(self.x_target[i-1]) == torch.Tensor:
            x_target = self.x_target[i-1]
            grad = omega_target * 2 * (self.x_seq[i] - x_target)
        return grad

    def partial_lu(self, i):
        # 目的関数に依存
        # grad = np.zeros(self.u_seq.shape[1])
        omega_u = 0.5
        omega_jerk = 10
        grad = omega_u * 2 * self.u_seq[i]
        if i == 0:
            # print("i-1:", self.u_seq[i-1])
            # print("i:", self.u_seq[i])
            grad += omega_jerk * 2 * (self.u_seq[i] - self.u_seq[i+1])
        elif i == self.u_seq.shape[0] - 1:
            grad += omega_jerk * 2 * (self.u_seq[i] - self.u_seq[i-1])
        else:
            grad += omega_jerk * 2 * (- self.u_seq[i+1] + 2*self.u_seq[i] - self.u_seq[i-1])
        return grad

    def compute_grad(self):
        grad_xu_mat = torch.zeros((self.seq_len, self.seq_len, self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                grad_xu_mat[i, j] = self.grad_xu(i+1, j)

        partial_lx = torch.stack([self.partial_lx(i+1) for i in range(self.seq_len)]).T
        partial_lu = torch.stack([self.partial_lu(i) for i in range(self.seq_len)])
        grad_u = torch.einsum('xi,ijxu->ju', partial_lx, grad_xu_mat) + partial_lu
        grad_u_clipped = torch.clamp(grad_u, -1, 1)
        return grad_u_clipped

    def optim(self, optim_itr = 5, eta=0.01, plot=False):
        if plot:#3d plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            x_seq = self.x_seq.detach().cpu().numpy()
            ax.plot(x_seq[:, 0], x_seq[:, 1], x_seq[:, 2], marker=",")
            # plot x_target
            colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
            for i in range(len(self.x_target)):
                if type(self.x_target[i]) == torch.Tensor:
                    target = self.x_target[i].detach().cpu().numpy()
                    ax.plot([target[0]], [target[1]], [target[2]], marker="x", color=colors[i % len(colors)])
                    ax.plot([x_seq[i][0]], [x_seq[i][1]], [x_seq[i][2]], marker="o", color=colors[i % len(colors)])
            # plt.show()
            
        X,Y,Z = [],[],[]

        for i in range(optim_itr):
            print("iteration:", i)
            cm = plt.get_cmap("Spectral")
            z = i/optim_itr
            grad = self.compute_grad()
                # print("grad:", grad)
            self.u_seq = self.u_seq - eta * grad
            self.compute_x_seq(self.u_seq)
            x_seq = self.x_seq.detach().cpu().numpy()
            X = np.append(X,[x_seq[:, 0]])
            Y = np.append(Y,[x_seq[:, 1]])
            Z = np.append(Z,[x_seq[:, 2]])
            
            X = X.reshape([i+1,x_seq.shape[0]])
            Y = Y.reshape([i+1,x_seq.shape[0]])
            Z = Z.reshape([i+1,x_seq.shape[0]])
        
            if plot:
                ax.plot(X[i], Y[i], Z[i], marker=",", color = cm(z)) 

        if plot:
            plt.show()        

        return self.u_seq

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    env = HoverAviary(act=ActionType.DYN)

    dynamics_model = GITAI(
        output_shape=env.observation_space.shape[0],
        input_shape=env.observation_space.shape[0]+env.action_space.shape[0],
        device=device)
    # print("input_shape:", env.observation_space.shape[0]+env.action_space.shape[0])
    # print("output_shape:", env.observation_space.shape[0])
    
    time_step = 50
    state_dim = 12 # x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz
    initial_x = np.zeros(state_dim)
    # initial_u_seq = np.concatenate([np.array([[1, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(int(time_step/2))]), np.array([[1, -0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(int(time_step/2))])]) # thrust, roll, pitch, yaw
    initial_u_seq = np.concatenate([np.array([[1, 0.2, 0.2, 0] for _ in range(int(time_step))])])
    model_based = True

    x_target = [None for _ in range(time_step)]
    x_target[10] = np.array([2, 3, -1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_target[20] = np.array([5, 5, -1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_target[25] = np.array([2, 7, 4,
    #                          0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_target[-1] = np.array([0, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    optim = Optimizer(
        initial_x=initial_x,
        u_seq=initial_u_seq,
        x_target=x_target,
        model=dynamics_model,
        device=device,
        model_based=model_based
        )
    optim.optim(plot=True)