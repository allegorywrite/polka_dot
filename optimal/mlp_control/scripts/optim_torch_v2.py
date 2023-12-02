import numpy as np
import torch
from optimal.mlp_control.models.gitai import GITAI
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

class Optimizer:

    def __init__(
            self, 
            initial_x, 
            u_seq, 
            x_target, 
            model=None, 
            model_based=False, 
            latent_dim=12,
            device=torch.device('cpu'),
            env=None,
            gui=False,):
        self.env = env
        self.gui = gui
        self.x_seq = torch.zeros((len(u_seq)+1, latent_dim)).to(device)
        initial_x_normalized = self.env._clipAndNormalizeState(initial_x)
        self.x_seq[0] = torch.from_numpy(initial_x_normalized[:latent_dim]).float().to(device)
        self.x_expert_seq = torch.zeros((len(u_seq)+1, initial_x.shape[0])).to(device)
        self.x_expert_seq[0] = torch.from_numpy(initial_x).float().to(device)
        self.u_seq = torch.from_numpy(u_seq).float().to(device)
        self.model = model
        self.device = device
        self.x_target = x_target
        self.x_target_normalized = [None for _ in range(len(x_target))]
        for i in range(len(x_target)):
            if x_target[i] is not None:
                x_target_normalized = self.env._clipAndNormalizeState(x_target[i])
                self.x_target_normalized[i] = torch.from_numpy(x_target_normalized[:9]).float().to(device)
        self.model_based = model_based
        self.seq_len = u_seq.shape[0]
        self.compute_x_seq(self.u_seq)

        # privisional
        self.xu_mat = torch.zeros((self.x_seq.shape[1], self.u_seq.shape[1])).to(self.device)
        self.xx_mat = torch.zeros((self.x_seq.shape[1], self.x_seq.shape[1])).to(self.device)
        self.x_mat = torch.zeros(self.x_seq.shape[1]).to(self.device)
        self.identity = torch.eye(self.x_seq.shape[1]).to(self.device)

        self.log_itr = 0

        self.omega_target = 100
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
            delta[3] = u[1]*0.1
            delta[4] = u[2]*0.1
            delta[5] = u[3]*0.1
        else:
            A = torch.zeros((x.shape[0], x.shape[0])).to(self.device)
            B = torch.zeros((x.shape[1], u.shape[1])).to(self.device)
            B[0,1] = 1
            B[1,2] = 1
            B[2,3] = 1
            B[3,1] = 0.1
            B[4,2] = 0.1
            B[5,3] = 0.1
            Bu = torch.mm(B, u.T).T
            delta = torch.mm(A, x) + Bu
        return delta
    
    def step(self, x, u):
        gain = 100
        if self.model_based:
            delta = self.model_based_model(x, u)
        else:
            if len(x.shape) == 1:
                input = torch.cat((x[3:], u))
            else:
                input = torch.cat((x[:, 3:], u), len(x.shape)-1)
            delta = self.model(input)
        if self.gui:
            print("obs:", x)
            print("action:", u)
            print("delta:", delta)
        return delta/gain

    def compute_x_seq(self, u_seq):
        start = time.time()
        self.env.reset()
        for i in range(self.seq_len):
            self.x_seq[i+1] = self.x_seq[i] + self.step(self.x_seq[i], u_seq[i])
            obs, _, _, info = self.env.step(u_seq[i].detach().cpu().numpy())
            # time.sleep(0.01)
            # print("obs:", obs[0:3])
            ret = np.hstack([info["raw_obs"][0:3], info["raw_obs"][7:10], info["raw_obs"][10:13], info["raw_obs"][13:16]]).reshape(12,)
            # print("ret:", ret)
            self.x_expert_seq[i+1] = torch.from_numpy(ret).float().to(self.device)
            # print("x_expert_seq:", self.x_expert_seq[i+1])

    def grad_xu(self, i, j):
        # grad = torch.zeros((self.x_seq.shape[1], self.u_seq.shape[1]))
        grad = self.xu_mat
        if j < i - 1:
            grad = torch.mm(self.grad_xx(i, i-1),self.grad_xu(i-1, j))
        elif j == i - 1:
            grad = self.grad_f(i-1)[1]
        return grad

    def grad_xx(self, i, j):
        # grad = torch.zeros((self.x_seq.shape[1], self.x_seq.shape[1]))
        grad = self.xx_mat
        identity = self.identity
        if j < i -1:
            grad = torch.dot(self.grad_xx(i, i-1),self.grad_xx(i-1, j))
        elif j == i - 1:
            grad = identity + self.grad_f(i-1)[0]
        elif j == i:
            grad = identity
        return grad
    
    def compute_grad_f(self, x_seq, u_seq, batch_size=100):
        x_seq_batch_list = torch.split(x_seq, batch_size)
        u_seq_batch_list = torch.split(u_seq, batch_size)
        
        self.x_grad_mat_all = torch.zeros((0, x_seq.shape[1], x_seq.shape[1])).to(self.device)
        self.u_grad_mat_all = torch.zeros((0, x_seq.shape[1], u_seq.shape[1])).to(self.device)
        for i in range(len(x_seq_batch_list)):
            x_seq_batch = x_seq_batch_list[i]
            u_seq_batch = u_seq_batch_list[i]
            x_seq_batch.requires_grad_(True)
            u_seq_batch.requires_grad_(True)
            delta = self.step(x_seq_batch, u_seq_batch)
            x_grad_mat = torch.zeros((delta.shape[0], x_seq_batch.shape[1], x_seq_batch.shape[1])).to(self.device)
            u_grad_mat = torch.zeros((delta.shape[0], x_seq_batch.shape[1], u_seq_batch.shape[1])).to(self.device)

            start = time.time()

            for j in range(delta.shape[0]):# batch_size
                for k in range(delta.shape[1]): # state_size
                    x_grad = torch.autograd.grad(delta[j, k], x_seq_batch, create_graph=True)[0]
                    x_grad_mat[j, k] = x_grad[j].detach()
                    u_grad = torch.autograd.grad(delta[j, k], u_seq_batch, create_graph=True)[0]
                    u_grad_mat[j, k] = u_grad[j].detach()
                    self.log_itr += 1
                    elapsed_time = time.time() - start
                    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            # print("count:", j*k)

            self.x_grad_mat_all = torch.cat((self.x_grad_mat_all, x_grad_mat), 0)
            self.u_grad_mat_all = torch.cat((self.u_grad_mat_all, u_grad_mat), 0)

    def grad_f(self, i):
        grad_x = self.x_grad_mat_all[i]
        grad_u = self.u_grad_mat_all[i]
        return grad_x, grad_u

    def partial_lx(self, i):
        # 目的関数に依存
        grad = self.x_mat
        if type(self.x_target_normalized[i-1]) == torch.Tensor:
            x_target = self.x_target_normalized[i-1]
            grad = self.omega_target * 2 * (self.x_seq[i] - x_target)
            grad[3:] = 0
        return grad

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
        cost_target = 0
        for i in range(len(self.x_target_normalized)):
            if type(self.x_target_normalized[i]) == torch.Tensor:
                diff_vector = self.x_target_normalized[i][0:3]-x_seq[i+1][0:3]
                # diff_vector[3:] = 0
                cost_target += torch.norm(diff_vector)**2

        total_cost = self.omega_u*cost_u + self.omega_jerk*cost_jark + self.omega_target*cost_target
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
    
    def plot_quiver(self, ax, x, y, z, roll, pitch, yaw):
        for i in range(len(x)):
            R = np.array([[np.cos(yaw[i])*np.cos(pitch[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.sin(roll[i])-np.sin(yaw[i])*np.cos(roll[i]), np.cos(yaw[i])*np.sin(pitch[i])*np.cos(roll[i])+np.sin(yaw[i])*np.sin(roll[i])],
                        [np.sin(yaw[i])*np.cos(pitch[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.sin(roll[i])+np.cos(yaw[i])*np.cos(roll[i]), np.sin(yaw[i])*np.sin(pitch[i])*np.cos(roll[i])-np.cos(yaw[i])*np.sin(roll[i])],
                        [-np.sin(pitch[i]), np.cos(pitch[i])*np.sin(roll[i]), np.cos(pitch[i])*np.cos(roll[i])]])
            v = np.dot(R, np.array([0, 0, 1])) 
            ax.quiver(x[i], y[i], z[i], v[0], v[1], v[2], length=0.03, normalize=True, color="r")

    def optim(self, optim_itr = 10, eta=0.1, plot=False, plot3d=False, feedback=False):
        x_seq = self.x_seq.detach().cpu().numpy()
        x_seq_expanded = self.env.expandState(x_seq)
        x_expert_seq = self.x_expert_seq.detach().cpu().numpy()
        if plot:#3d plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # model prediction
            ax.plot(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], marker=",")
            self.plot_quiver(ax, x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], x_seq_expanded[:, 3], x_seq_expanded[:, 4], x_seq_expanded[:, 5])
            # expert trajectory
            ax.plot(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], marker=",", color="g")
            self.plot_quiver(ax, x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], x_expert_seq[:, 3], x_expert_seq[:, 4], x_expert_seq[:, 5])

            ax.plot([x_seq_expanded[0, 0]], [x_seq_expanded[0, 1]], [x_seq_expanded[0, 2]], marker="o", color="r")
            # plot x_target
            colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
            for i in range(len(self.x_target)):
                # print(self.x_target[i], type(self.x_target[i]))
                if type(self.x_target[i]) == list or type(self.x_target[i]) == np.ndarray:
                    target = self.x_target[i]
                    ax.plot([target[0]], [target[1]], [target[2]], marker="x", color=colors[i % len(colors)])
                    ax.plot([x_seq_expanded[i][0]], [x_seq_expanded[i][1]], [x_seq_expanded[i][2]], marker="o", color=colors[i % len(colors)])
            # ax.set_xlim([-1, 1])
            # ax.set_ylim([-1, 1])
            # ax.set_zlim([0, 2])
            # plt.show()

        # if plot:
        #     o3d_vis = o3d.visualization.Visualizer()
        #     o3d_vis.create_window(window_name='3D Viewer', width=400, height=300, visible=True)
        #     lineset = o3d.geometry.LineSet()
        #     lineset.points = o3d.utility.Vector3dVector(np.array([x_seq[:, 0], x_seq[:, 1], x_seq[:, 2]]).T)
        #     lineset.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(x_seq.shape[0]-1)]))
        #     lineset.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for i in range(x_seq.shape[0]-1)]))
        #     o3d_vis.add_geometry(lineset)

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
           
            
        X,Y,Z = [],[],[]

        for i in range(optim_itr):
            print("iteration:", i)
            cm = plt.get_cmap("Spectral")
            z = i/optim_itr
            self.compute_grad_f(self.x_seq[:-1], self.u_seq)
            grad = self.compute_grad()
            self.u_seq = self.u_seq - eta * grad

            self.compute_x_seq(self.u_seq)
            self.compute_x_seq(self.u_seq)
            cost = self.compute_cost(self.x_seq)
            print("model cost:", cost)
            cost = self.compute_cost(self.x_expert_seq)
            print("real cost:", cost)

            self.x_seq = self.x_seq.detach()
            self.u_seq = self.u_seq.detach()
            
            x_seq = self.x_seq.detach().cpu().numpy()
            x_seq_expanded = self.env.expandState(x_seq)
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
            ax.plot([x_seq_expanded[-1, 0]], [x_seq_expanded[-1, 1]], [x_seq_expanded[-1, 2]], marker="o", color="r")
            ax.plot(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], marker=",", color="r")
            # ax.set_xlim([-0.1, 0.1])
            # ax.set_ylim([-0.1, 0.1])
            # ax.set_zlim([0, 0.2])
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 2])
            plt.show()   

        if plot3d:
            self.sim_manager.draw_trajectory(x_expert_seq[:, 0], x_expert_seq[:, 1], x_expert_seq[:, 2], eular=x_expert_seq[:, 3:6], color=[0.3, 0.8, 1.0])
            # self.sim_manager.draw_trajectory(x_seq_expanded[:, 0], x_seq_expanded[:, 1], x_seq_expanded[:, 2], eular=x_seq_expanded[:, 3:6], color=[1, 0.5, 0])
            self.sim_manager.render_field()
            self.sim_manager.viz_run()  

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
    state_dim = 12 # x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz
    latent_dim = 9 # x, y, z, roll, pitch, yaw, vx, vy, vz
    initial_x = np.zeros(state_dim)
    initial_x[2] = 1
    # initial_u_seq = np.concatenate([np.array([[1, 0.2, 0.2, 0] for _ in range(int(time_step/2))]), np.array([[1, -0.2, 0.2, 0] for _ in range(int(time_step/2))])]) # thrust, roll, pitch, yaw
    # initial_u_seq = np.concatenate([np.array([[0.5, 0, -0.5, 0] for _ in range(int(time_step))])])
    initial_u_seq = np.zeros((time_step, 4))
    initial_u_seq[:int(time_step/2)] = np.concatenate([np.array([[1, 0, 0, 0] for _ in range(int(time_step/2))])])
    initial_u_seq[int(time_step/2):] = np.concatenate([np.array([[1, 0, 0, 0] for _ in range(int(time_step/2))])])
    model_based = False

    DEFAULT_GUI = False
    DEFAULT_RECORD_VIDEO = False
    DEFAULT_OUTPUT_FOLDER = 'results'
    DEFAULT_COLAB = False
    DEFAULT_SIMULATION_FREQ_HZ = 240
    DEFAULT_CONTROL_FREQ_HZ = 60
    AGGR_PHY_STEPS = int(DEFAULT_SIMULATION_FREQ_HZ/DEFAULT_CONTROL_FREQ_HZ)

    env = HoverAviary(
        gui=DEFAULT_GUI,
        act=ActionType.DYN,
        initial_xyzs=np.array([initial_x[:3]]),
        initial_rpys=np.array([initial_x[3:6]]),
        freq=DEFAULT_SIMULATION_FREQ_HZ,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        )

    dynamics_model = GITAI(
        output_shape=env.observation_space.shape[0]-3,
        input_shape=env.observation_space.shape[0]-6+env.action_space.shape[0],
        device=device)
    dynamics_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../results/dynamics_model.pt")))
    # print("input_shape:", env.observation_space.shape[0]+env.action_space.shape[0])
    # print("output_shape:", env.observation_space.shape[0])

    x_target = [None for _ in range(time_step)]
    # x_target[10] = np.array([0.2, 0.3, -0.1,
    #                          0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_target[20] = np.array([0.5, 0.3, -0.1,
    #                          0, 0, 0, 0, 0, 0, 0, 0, 0])
    # x_target[-1] = np.array([0.5, 0, 1.3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_target[-1] = np.array([0.5, -0.4, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    optim = Optimizer(
        initial_x=initial_x,
        u_seq=initial_u_seq,
        x_target=x_target,
        model=dynamics_model,
        device=device,
        model_based=model_based,
        latent_dim=latent_dim,
        env=env,
        gui=args.gui,
        )
    start = time.time()
    optim.optim(
        optim_itr=args.optim_itr,
        plot=args.matplotlib, 
        plot3d=args.open3d,
        feedback=args.feedback,
        eta=1.5
    )
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")