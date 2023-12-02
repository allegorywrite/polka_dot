import numpy as np
import torch
# from optimal.models.gitai import GITAI
from optimal.mlp_control.models.gitai import GITAI
from optimal.lie_control.models.SE3FVIN import SE3FVIN
from optimal.lie_control.systems.sim import SimulationManager
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
import matplotlib.pyplot as plt
import time
import os
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import open3d as o3d
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
import torch.optim as optim
from torchviz import make_dot
from IPython.display import display
import math
torch.set_default_dtype(torch.float32)

class Optimizer:

    def __init__(
            self, 
            initial_x, 
            u_seq, 
            x_target, 
            model=None,
            blackbox_model=None, 
            model_based=False, 
            latent_dim=12,
            device=torch.device('cpu'),
            env=None,
            gui=False,
            use_lie=False):
        self.env = env
        self.gui = gui
        self.latent_dim = latent_dim
        self.x_seq = torch.zeros((len(u_seq)+1, latent_dim)).to(device)
        # initial_x_normalized = self.env._clipAndNormalizeState(initial_x)
        self.initial_x = initial_x
        self.x_seq[0] = torch.from_numpy(initial_x[:latent_dim]).float().to(device)
        self.x_expert_seq = torch.zeros((len(u_seq)+1, initial_x.shape[0])).to(device)
        self.x_expert_seq[0] = torch.from_numpy(initial_x).float().to(device)
        self.x_real_seq = torch.zeros((len(u_seq)+1, initial_x.shape[0])).to(device)
        self.x_real_seq[0] = torch.from_numpy(initial_x).float().to(device)
        self.u_seq = torch.from_numpy(u_seq).float().to(device)
        self.model = model
        self.blackbox_model = blackbox_model
        self.device = device
        self.model_based = model_based
        self.seq_len = u_seq.shape[0]
        self.use_lie = use_lie

        self.x_target = x_target
        self.x_target_normalized = [None for _ in range(len(x_target))]
        for i in range(len(x_target)):
            if x_target[i] is not None:
                # x_target_normalized = self.env._clipAndNormalizeState(x_target[i])
                # self.x_target[i][3:] = self.x_seq[i].detach().cpu().numpy()[3:]
                self.x_target_normalized[i] = torch.from_numpy(x_target[i][:latent_dim]).float().to(device)

        self.compute_x_seq(self.u_seq)
                
        # privisional
        self.xu_mat = torch.zeros((self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        self.xx_mat = torch.zeros((self.x_seq.shape[1], self.x_seq.shape[1])).to(self.device)
        self.x_mat = torch.zeros(self.x_seq.shape[1]).to(self.device)
        self.identity = torch.eye(self.x_seq.shape[1]).to(self.device)

        self.log_itr = 0

        self.omega_target_pos = 5
        self.omega_target_att = 1
        self.omega_u = 0
        self.omega_jerk = 0
        self.sim_manager = SimulationManager()
        cost = self.compute_cost(self.x_seq)
        print("model cost:", cost)
        cost = self.compute_cost(self.x_expert_seq)
        print("real cost:", cost)

    # (roll pitch yaw)がそれぞれ(x, y, z)の微分値であるようなモデルを考える
    def model_based_model(self, x, u): # tensor -> tensor
        delta = torch.zeros(x.shape).to(self.device)
        if len(x.shape) == 1:
            delta[0] = u[1]
            delta[1] = u[2]
            delta[2] = u[3]
            # delta[3] = u[1]*0.1
            # delta[4] = u[2]*0.1
            # delta[5] = u[3]*0.1
        else:
            A = torch.zeros((x.shape[0], x.shape[0])).to(self.device)
            B = torch.zeros((x.shape[1], u.shape[1])).to(self.device)
            B[0,1] = 1
            B[1,2] = 1
            B[2,3] = 1
            B = B.to(dtype=torch.float32) 
            u = u.to(dtype=torch.float32)
            Bu = torch.mm(B, u.T).T
            delta = torch.mm(A, x) + Bu
        return delta
    
    def step(self, x, u):
        
        if self.model_based:
            x_next = x + self.model_based_model(x, u)
            return x_next

        input = torch.cat((x, u), len(x.shape)-1)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
            input.requires_grad_(True)
            x_next = self.model.forward_trainingd(input).squeeze(0)[:self.latent_dim]
        else:
            input.requires_grad_(True)
            x_next = self.model.forward_trainingd(input)[:,:self.latent_dim]

        # if self.gui:
        #     print("obs:", x)
        #     print("action:", u)
        #     print("x_next:", x_next)

        return x_next
    
    def compute_x_seq(self, u_seq):
        start = time.time()
        self.env.reset()
        # print("x_seq:", self.x_seq)
        for i in range(self.seq_len):
            input_aug = u_seq[i]*torch.Tensor([1.3,1.3,1.3,1.3]).to(self.device)
            self.x_seq[i+1] = self.step(self.x_seq[i], u_seq[i])
            # print("x_seq:", self.x_seq[i+1])
            obs, _, _, info = self.env.step(u_seq[i].detach().cpu().numpy())
            if self.gui:
                for j in range(len(self.x_target)):
                    if type(self.x_target[j]) == list or type(self.x_target[j]) == np.ndarray:
                        yaw = math.atan2(self.x_target[j][1]-self.initial_x[1], self.x_target[j][0]-self.initial_x[0])+math.pi/2
                        eular = [self.x_target[j][3], self.x_target[j][4], yaw]
                        env.addGate(list(self.x_target[j][:3]), eular)
                time.sleep(0.001)
            # print("obs:", obs[0:3])
            quat = info["raw_obs"][3:7]
            R = Rotation.from_quat(quat)
            # rotmat = R.as_matrix()
            eular = R.as_euler("xyz", degrees=False)
            ret = np.hstack([info["raw_obs"][0:3], eular, info["raw_obs"][10:13], info["raw_obs"][13:16]]).reshape(self.x_expert_seq.shape[1],)
            self.x_expert_seq[i+1] = torch.from_numpy(ret).float().to(self.device)
            # print("x_expert_seq:", self.x_expert_seq[i+1])

    def compute_cost_fb(self, x_pred, x_real_hat):
        diff = (x_pred - x_real_hat).T
        diff[:,:-1] = 0
        cost = torch.norm(x_pred - x_real_hat)**2
        return cost
    
    def feedback(self, x_seq, u_seq, x_cur, step=1, eta=1, itr=0):
        u_seq = u_seq.detach()
        x_pred = x_seq[itr+step+1:itr+2*step+1]
        x_real_hat = torch.zeros((2*step+1, latent_dim)).to(device)
        x_real_hat[0] = x_cur.detach()
        x_correct = torch.zeros((2*step+1, latent_dim)).to(device)
        x_correct[0] = x_cur.detach()
        for i in range(2*step):
            x_real_hat[i+1] = self.step(x_real_hat[i], u_seq[itr+i])

        print("cost_a:", self.compute_cost_fb(x_pred, x_real_hat[step+1:]))
        self.compute_grad_f_batch(
            x_real_hat[step+1:], 
            u_seq[itr+step:itr+2*step]
        )
        grad = self.compute_grad_fb(
            step=step, 
            x_pred=x_pred, 
            x_real_hat=x_real_hat[step+1:]
        )
        u_seq[itr+step:itr+2*step] = u_seq[itr+step:itr+2*step] - eta * grad
        for i in range(2*step):
            x_correct[i+1] = self.step(x_correct[i], u_seq[itr+i])

        print("cost_b:", self.compute_cost_fb(x_pred, x_correct[step+1:]))

        return u_seq, x_real_hat, x_correct
    
    def compute_grad_fb(self, step, x_pred, x_real_hat):
        grad_xu_mat = torch.zeros((step, step, self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        for i in range(step):
            for j in range(step):
                grad_xu_mat[i, j] = self.grad_xu(i, j)
        partial_lx = 2*(x_real_hat - x_pred).T
        partial_lx[:,:-1] = 0
        grad_u = torch.einsum('xi,ijxu->ju', partial_lx, grad_xu_mat)
        return grad_u
            
    def grad_xu(self, i, j):
        grad = self.xu_mat
        if j < i - 1:
            grad = torch.mm(self.grad_xx(i, i-1),self.grad_xu(i-1, j))
        elif j == i - 1:
            grad = self.grad_f(i-1)[1]
        return grad

    def grad_xx(self, i, j):
        grad = self.xx_mat
        identity = self.identity
        if j < i -1:
            grad = torch.dot(self.grad_xx(i, i-1),self.grad_xx(i-1, j))
        elif j == i - 1:
            grad = self.grad_f(i-1)[0]
        elif j == i:
            grad = identity
        return grad
    
    def compute_grad_f_batch(self, x_seq, u_seq, batch_size=100):
        x_seq_batch_list = torch.split(x_seq, batch_size)
        u_seq_batch_list = torch.split(u_seq, batch_size)
        
        self.x_grad_mat_all = torch.zeros((0, x_seq.shape[1], x_seq.shape[1])).to(self.device)
        self.u_grad_mat_all = torch.zeros((0, x_seq.shape[1], u_seq.shape[1])).to(self.device)

        for i in range(len(x_seq_batch_list)):
            x_seq_batch = x_seq_batch_list[i]
            u_seq_batch = u_seq_batch_list[i]
            x_seq_batch.requires_grad_()
            u_seq_batch.requires_grad_()
            x_next = self.step(x_seq_batch, u_seq_batch)
            x_grad_mat = torch.zeros((x_next.shape[0], x_seq_batch.shape[1], x_seq_batch.shape[1])).to(self.device)
            u_grad_mat = torch.zeros((x_next.shape[0], x_seq_batch.shape[1], u_seq_batch.shape[1])).to(self.device)

            start = time.time()
            time_list = []

            for j in range(x_next.shape[0]):# batch_size
                for k in range(x_next.shape[1]): # state_size
                    x_grad = torch.autograd.grad(x_next[j, k], x_seq_batch, create_graph=True)[0]
                    x_grad_mat[j, k] = x_grad[j]
                    u_grad = torch.autograd.grad(x_next[j, k], u_seq_batch, create_graph=True)[0]
                    u_grad_mat[j, k] = u_grad[j]
                    self.log_itr += 1
                    elapsed_time = time.time() - start
                    time_list.append(elapsed_time)
                    
                    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            # print("average time:", sum(time_list)/len(time_list))
            # print("count:", len(time_list))

            self.x_grad_mat_all = torch.cat((self.x_grad_mat_all, x_grad_mat), 0)
            self.u_grad_mat_all = torch.cat((self.u_grad_mat_all, u_grad_mat), 0)

        

    def grad_f(self, i):
        grad_x = self.x_grad_mat_all[i]
        grad_u = self.u_grad_mat_all[i]
        return grad_x, grad_u

    def partial_lx(self, i):
        # 目的関数に依存
        # grad = self.x_mat
        grad = torch.zeros(self.x_seq.shape[1]).to(self.device)
        if type(self.x_target_normalized[i-1]) == torch.Tensor:
            x_target = self.x_target_normalized[i-1]
            grad[0:3] = self.omega_target_pos * 2 * (self.x_seq[i][0:3] - x_target[0:3])
            grad[3:6] = self.omega_target_att * self.partial_Deular(self.x_seq[i][3:6], x_target[3:6])
            # grad[3:6] = grad[3:6] * self.omega_target_att/self.omega_target_pos
            # grad[6:] = 0
            # print("grad:", grad)    
        return grad
    
    def get_Rz(self, eular_target):
        x = eular_target[0]
        y = eular_target[1]
        z = eular_target[2]
        cosx = torch.cos(x)
        sinx = torch.sin(x)
        cosy = torch.cos(y)
        siny = torch.sin(y)
        cosz = torch.cos(z)
        sinz = torch.sin(z)
        Rz = torch.zeros((3)).to(self.device)
        Rz[0] = sinz * sinx + cosz * cosx * siny
        Rz[1] = cosx * sinz * siny - cosz * sinx
        Rz[2] = cosy * cosx
        return Rz
    
    def partial_Deular(self, eular, eular_target):
        Rdz = self.get_Rz(eular_target)
        psi = eular[0]
        theta = eular[1]
        phi = eular[2]
        cospsi = torch.cos(psi)
        sinpsi = torch.sin(psi)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        grad = torch.zeros((3)).to(self.device)
        grad[0] = Rdz[0]*(-sinpsi*sintheta*cosphi+cospsi*sinphi) + Rdz[1]*(-sinpsi*sintheta*sinphi-cospsi*cosphi) + Rdz[2]*(-sinpsi*costheta)
        grad[1] = Rdz[0]*(cospsi+costheta*cosphi) + Rdz[1]*(cospsi*costheta*sinphi) + Rdz[2]*(-cospsi*sintheta)
        grad[2] = Rdz[0]*(-cospsi*sintheta*sinphi-sinpsi*cosphi) + Rdz[1]*(cospsi*sintheta*cosphi+sinpsi*sintheta)
        return -grad
    
    def partial_lu(self, i):
        # 目的関数に依存
        grad = self.omega_u * 2 * self.u_seq[i]
        if i == 0:
            grad += self.omega_jerk * 2 * (self.u_seq[i] - self.u_seq[i+1])
        elif i == self.u_seq.shape[0] - 1:
            grad += self.omega_jerk * 2 * (self.u_seq[i] - self.u_seq[i-1])
        else:
            grad += self.omega_jerk * 2 * (- self.u_seq[i+1] + 2*self.u_seq[i] - self.u_seq[i-1])
        return grad
    
    def compute_cost(self, x_seq):
        cost_u = torch.norm(self.u_seq)**2
        cost_jark = sum(torch.norm(self.u_seq[i] - self.u_seq[i-1])**2 for i in range(1, len(self.u_seq)))
        cost_target_pos = 0
        cost_target_att = 0
        for i in range(len(self.x_target_normalized)):
            if type(self.x_target_normalized[i]) == torch.Tensor:
                pos_diff = self.x_target_normalized[i][0:3]-x_seq[i+1][0:3]
                att_diff = torch.dot(self.get_Rz(self.x_target_normalized[i][3:6]), self.get_Rz(x_seq[i+1][3:6]))
                # pos_diff = self.x_target_normalized[i]-self.x_seq[i+1]
                # pos_diff[3:] = 0
                cost_target_pos += self.omega_target_pos*torch.norm(pos_diff)**2
                cost_target_att -= self.omega_target_att*att_diff

        total_cost = self.omega_u*cost_u + self.omega_jerk*cost_jark + self.omega_target_pos*cost_target_pos+self.omega_target_att*cost_target_att
        # total_cost = self.omega_u*cost_u + self.omega_jerk*cost_jark + self.omega_target_pos*cost_target_pos

        return total_cost

    def compute_grad(self):
        grad_xu_mat = torch.zeros((self.seq_len, self.seq_len, self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                grad_xu_mat[i, j] = self.grad_xu(i+1, j)

        partial_lx = torch.stack([self.partial_lx(i+1) for i in range(self.seq_len)]).T
        partial_lu = torch.stack([self.partial_lu(i) for i in range(self.seq_len)])
        grad_u = torch.einsum('xi,ijxu->ju', partial_lx, grad_xu_mat) + partial_lu
        grad_u_clipped = torch.clamp(grad_u, -1, 1)
        # return grad_u_clipped
        return grad_u
    
    def plot_quiver(self, ax, x, y, z, R_mat):
        for i in range(len(x)):
            # R = np.array([[np.cos(yaw[i])*np.cos(pitch[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.sin(roll[i])-np.sin(yaw[i])*np.cos(roll[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.cos(roll[i])+np.sin(yaw[i])*np.sin(roll[i])],
            #             [np.sin(yaw[i])*np.cos(pitch[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.sin(roll[i])+np.cos(yaw[i])*np.cos(roll[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.cos(roll[i])-np.cos(yaw[i])*np.sin(roll[i])],
            #             [-np.sin(pitch[i]), np.cos(pitch[i])*np.sin(roll[i]), np.cos(pitch[i])*np.cos(roll[i])]])
            R = R_mat[i].reshape(3, 3)
            v = np.dot(R, np.array([0, 0, 1])) 
            ax.quiver(x[i], y[i], z[i], v[0], v[1], v[2], length=0.03, normalize=True, color="r")

    def optim(self, optim_itr = 1, eta=1, plot=False, plot3d=False, feedback=False):
        x_seq = self.x_seq.detach().cpu().numpy()
        # x_seq_expanded = self.env.expandState(x_seq)
        x_seq_expanded = x_seq
        x_expert_seq = self.x_expert_seq.detach().cpu().numpy()
        if plot:#3d plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # model prediction
            ax.plot(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], marker=",")
            self.plot_quiver(ax, x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], x_seq_expanded[:, 3:6])
            # expert trajectory
            ax.plot(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], marker=",", color="g")
            self.plot_quiver(ax, x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], x_expert_seq[:, 3:6])

            ax.plot([x_seq_expanded[0, 0]], [x_seq_expanded[0, 1]], [x_seq_expanded[0, 2]], marker="o", color="r")
            # plot x_target
            colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
            for i in range(len(self.x_target)):
                if type(self.x_target[i]) == list or type(self.x_target[i]) == np.ndarray:
                    target = self.x_target[i]
                    ax.plot([target[0]], [target[1]], [target[2]], marker="x", color=colors[i % len(colors)])
                    ax.plot([x_seq_expanded[i][0]], [x_seq_expanded[i][1]], [x_seq_expanded[i][2]], marker="o", color=colors[i % len(colors)])
        
        if plot3d:
            self.sim_manager.create_field()
            self.sim_manager.add_origin()
            # self.sim_manager.add_point(target[0], target[1], target[2])
            self.sim_manager.draw_trajectory(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], eular=x_expert_seq[:, 3:6], color=[0.7, 0.7, 0.7])
            # self.sim_manager.draw_trajectory(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], eular=x_seq_expanded[:, 3:6], color=[0.7, 0.7, 0.7])

            for i in range(len(self.x_target)):
                if type(self.x_target[i]) == list or type(self.x_target[i]) == np.ndarray:
                    target = self.x_target[i]
                    # self.sim_manager.add_point(target[0], target[1], target[2])
                    self.sim_manager.add_quadcopter(target[0], target[1], target[2], eular=target[3:6], color=[0, 1, 0])
                    # self.sim_manager.add_point(x_seq_expanded[i][0], x_seq_expanded[i][1], x_seq_expanded[i][2])
            self.sim_manager.render_field()

        optimizer = optim.Adam([self.u_seq], lr=eta)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optim_itr)         
            
        X,Y,Z = [],[],[]

        for i in range(optim_itr):
            print("iteration:", i)
            cm = plt.get_cmap("Spectral")
            z = i/optim_itr
            
            self.compute_grad_f_batch(self.x_seq[:-1], self.u_seq)
            
            grad = self.compute_grad()
            self.u_seq = self.u_seq - eta * grad

            # optimizer.zero_grad()
            # self.u_seq.grad = grad
            # optimizer.step()
            
            self.compute_x_seq(self.u_seq)
            cost = self.compute_cost(self.x_seq)
            print("model cost:", cost)
            cost = self.compute_cost(self.x_expert_seq)
            print("real cost:", cost)
            # scheduler.step()
            # current_lr = scheduler.get_last_lr()[0]
            # print("current learning rate:", current_lr)
            
            # self.x_seqとself.u_seqを計算グラフから切り離す
            self.x_seq = self.x_seq.detach()
            self.u_seq = self.u_seq.detach()

            # x_seq_expanded = self.env.expandState(x_seq)
            x_seq = self.x_seq.detach().cpu().numpy()
            x_seq_expanded = x_seq
            X = np.append(X,[x_seq_expanded[:, 0]])
            Y = np.append(Y,[x_seq_expanded[:, 1]])
            Z = np.append(Z,[x_seq_expanded[:, 2]])
            
            X = X.reshape([i+1,x_seq_expanded.shape[0]])
            Y = Y.reshape([i+1,x_seq_expanded.shape[0]])
            Z = Z.reshape([i+1,x_seq_expanded.shape[0]])
        
            if plot:
                ax.plot(X[i], Y[i], Z[i], marker=",", color = cm(z)) 

            # if plot3d:
            #     self.sim_manager.draw_trajectory(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], eular=x_seq_expanded[:, 3:6], color=[z, z, 1-z])
            #     self.sim_manager.render_field()

        print("log_itr:", self.log_itr)

        x_expert_seq = self.x_expert_seq.detach().cpu().numpy()
        if plot:
            ax.plot(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], marker=",", color="r")
            ax.plot([x_seq_expanded[-1, 0]], [x_seq_expanded[-1, 1]], [x_seq_expanded[-1, 2]], marker="o", color="r")
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 2])
            plt.show()
        if plot3d:
            # self.sim_manager.draw_trajectory(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], eular=x_seq_expanded[:, 3:6], color=[0.9, 0.8, 0.2])#0.8, 0.8, 0.4
            self.sim_manager.draw_trajectory(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], eular=x_expert_seq[:, 3:6], color=[0.9, 0, 0])
            self.sim_manager.render_field()
            self.sim_manager.viz_run()

        if feedback:
            self.sim_traj(self.x_seq, self.u_seq)   

        # if plot:
        #     o3d_vis.run()   

        return self.u_seq

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='debug mode')
    parser.add_argument('--open3d', action='store_true', help='debug mode')
    parser.add_argument('--matplotlib', action='store_true', help='debug mode')
    parser.add_argument('--feedback', action='store_true', help='debug mode')
    parser.add_argument('--optim_itr', type=int, default=0, help='debug mode')
    parser.add_argument('--blackbox', action='store_true', help='debug mode')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    time_step = 32
    state_dim = 12 # x, y, z, omega, vx, vy, vz, wx, wy, wz
    latent_dim = 12 # x, y, z, omega, vx, vy, vz, wx, wy, wz
    initial_x = np.zeros(state_dim)
    initial_x[2] = 1
    initial_x[3:6] = [0, 0, 0]
    initial_u_seq = np.zeros((time_step, 4))
    initial_u_seq[:int(time_step/2)] = np.concatenate([np.array([[0.5, 0, 0, 0] for _ in range(int(time_step/2))])])
    initial_u_seq[int(time_step/2):] = np.concatenate([np.array([[-1, 0, 0, 0] for _ in range(int(time_step/2))])])
    model_based = False

    DEFAULT_SIMULATION_FREQ_HZ = 240
    DEFAULT_CONTROL_FREQ_HZ = 60
    AGGR_PHY_STEPS = int(DEFAULT_SIMULATION_FREQ_HZ/DEFAULT_CONTROL_FREQ_HZ)

    env = HoverAviary(
        gui=args.gui,
        act=ActionType.DYN,
        initial_xyzs=np.array([initial_x[:3]]),
        initial_rpys=np.array([initial_x[3:6]]),
        freq=DEFAULT_SIMULATION_FREQ_HZ,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        )
    
    dt = 1.0 / DEFAULT_CONTROL_FREQ_HZ
    dynamics_model = SE3FVIN(device=device, time_step=dt).to(device)
    path = os.path.join(os.path.dirname(__file__), "../data/dynamics_model.pth")
    dynamics_model.load_state_dict(torch.load(path, map_location=device))

    blackbox_model = GITAI(
        output_shape=env.observation_space.shape[0]-3,
        input_shape=env.observation_space.shape[0]-6+env.action_space.shape[0],
        device=device)
    blackbox_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../data/blackbox.pth")))
    # blackbox_model.eval()


    x_target = [None for _ in range(time_step)]
    x_target[-1] = np.array([0.5, -0.4, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_target[15] = np.array([0.25, -0.2, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_target[-1] = np.array([0.5, -0.4, 1.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    env.addGate(list(x_target[-1][:3]), list(x_target[-1][3:6]))
    traj_optimizer = Optimizer(
        initial_x=initial_x,
        u_seq=initial_u_seq,
        x_target=x_target,
        model=dynamics_model,
        blackbox_model=blackbox_model,
        device=device,
        model_based=model_based,
        latent_dim=latent_dim,
        env=env,
        gui=args.gui,
        use_lie=(not args.blackbox)
        )
    start = time.time()
    traj_optimizer.optim(
        optim_itr=args.optim_itr,
        plot=args.matplotlib, 
        plot3d=args.open3d,
        feedback=args.feedback,
        eta=2.2
    )
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")