import glob
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count
# from torch import nn, tanh, relu
import sys
from pathlib import Path
import resource
# import torch
import random 
import os
import open3d as o3d
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

sys.path.append(str(Path(__file__).parent.parent))

class DataAnalyzer:
    def __init__(self, visualize=False):
      # if torch.cuda.is_available():
      #   self.device = torch.device('cuda')
      # else:
      #   self.device = torch.device('cpu')
      self.drones_num = 10
      self.replay_dir = "../data/training/replay/agents{}_*.npy".format(self.drones_num)
      self.train_dataset = []
      self.visualize = visualize
      self.neighbor_data = np.empty((0, 3))
      self.data_num_max = 100000

    def generate_dataset(self, replay_file):
      replay_data = np.load(replay_file, allow_pickle=True)
      map_file = "{}/../map/{}".format(os.path.dirname(replay_file), os.path.basename(replay_file))
      map_data = np.load(map_file, allow_pickle=True)
      print("replay_data.shape = {}, map_data.shape = {}".format(replay_data.shape, map_data.shape))
      for t in range(0, replay_data.shape[0]-1):
        if replay_data[t,0] == map_data[t,0]:
          point_clouds = map_data[t,1]
          log_time = replay_data[t,0]
          for i in range(0, replay_data.shape[1]-1):
            p_i_world = np.array(replay_data[t,i+1][0:3])
            p_next = np.array(replay_data[t+1,i+1][0:3])
            q_i_world = np.quaternion(replay_data[t,i+1][3], replay_data[t,i+1][4], replay_data[t,i+1][5], replay_data[t,i+1][6])
            v_i_world = np.array(replay_data[t,i+1][7:10])
            w_i_world = np.array(replay_data[t,i+1][10:13])
            observation_world = point_clouds[i]
            # 回転行列の計算
            R_inverse_iw = quaternion.as_rotation_matrix(q_i_world.conjugate())
            # オイラー角の計算
            euler_iw = quaternion.as_euler_angles(q_i_world.conjugate())
            # listデータをopen3dのPointCloudに変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(observation_world)
            # 座標変換
            pcd.translate(-p_i_world)
            pcd.rotate(R_inverse_iw)
            # 座標変換後の点群をlistに変換
            observation_local = np.asarray(pcd.points)
            # アクション
            p_next_local = np.dot(R_inverse_iw, p_next - p_i_world)

            for j in range(0, replay_data.shape[1]-1):
              if i != j:
                p_j_world = np.array(replay_data[t,j+1][0:3])
                v_j_world = np.array(replay_data[t,j+1][7:10])
                p_ij_local = np.dot(R_inverse_iw, p_j_world - p_i_world)
                v_ij_local = np.dot(R_inverse_iw, v_j_world - v_i_world) + np.cross(w_i_world, p_ij_local)
                self.neighbor_data = np.concatenate((self.neighbor_data, p_ij_local.reshape(1,-1), v_ij_local.reshape(1,-1)), axis=0)
                
            if(self.visualize and t == 0 and i == 0):
              self.visualize_data(replay_data, map_data[t,1], self.neighbor_data, observation_local, p_next_local)

            # データセットの作成(TODO)
            dataset = []

            self.neighbor_data = np.empty((0, 3))
        else:
          print("time sync error")
      
      return dataset

    def load_data(self):
      files = glob.glob(self.replay_dir)
      len_case = 0
      if self.visualize:
        self.generate_dataset(files[0])
        self.visualize = False
      with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        itr = 0
        for dataset in executor.map(self.generate_dataset, files):
          len_case += len(dataset)
          itr += 1
          print('files = {}, len_case = {}'.format(itr, len_case))
          if len_case > self.data_num_max:
            break
          self.train_dataset.extend(dataset)
      print('Total Training Dataset Size: ',len(self.train_dataset))

    def visualize_data(self, replay_data, point_clouds_world, replay_data_local, point_cloud_local, p_next_local):
      # ワールド座標系
      fig = plt.figure()
      ax = fig.add_subplot(111, projection="3d")
      ax.set_xlabel("x")
      ax.set_ylabel("y")
      ax.set_zlabel("z")
      colors = ["r", "g", "b", "c", "m", "y", "k"]
      for agent_id in range(0, replay_data.shape[1]-1):
        # エージェントの軌跡の描画
        trajectory_x = []
        trajectory_y = []
        trajectory_z = []
        for t in range(0, replay_data.shape[0]):
            trajectory_x.append(replay_data[t,agent_id+1][0])
            trajectory_y.append(replay_data[t,agent_id+1][1])
            trajectory_z.append(replay_data[t,agent_id+1][2])
        ax.plot(trajectory_x, trajectory_y, trajectory_z, 
                color=colors[agent_id % len(colors)], label="drone{}".format(agent_id))
        # 点群の描画
        point_cloud = np.array(point_clouds_world[agent_id])
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], 
                    color=colors[agent_id % len(colors)], s=1)
      plt.legend()

      # ローカル座標系(t=0)
      fig2 = plt.figure()
      ax2 = fig2.add_subplot(111, projection="3d")
      ax2.set_xlabel("x")
      ax2.set_ylabel("y")
      ax2.set_zlabel("z")
      for agent_id in range(0, int(replay_data_local.shape[0]/2)):
        neighbor_pos = replay_data_local[agent_id*2] # x, y, z
        neighbor_vel = replay_data_local[agent_id*2+1] # vx, vy, vz
        ax2.scatter(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2], 
                    color=colors[agent_id % len(colors)], s=10)
        ax2.quiver(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2], 
                    neighbor_vel[0], neighbor_vel[1], neighbor_vel[2], 
                    color=colors[agent_id % len(colors)], length=0.5, normalize=True)
      ax.scatter(point_cloud_local[:,0], point_cloud_local[:,1], point_cloud_local[:,2], 
                    color=colors[agent_id % len(colors)], s=1)
      ax2.quiver(0, 0, 0, p_next_local[0], p_next_local[1], p_next_local[2], 
                    color="black", length=0.5, normalize=True)
      plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--visualize", action="store_true")
  args = parser.parse_args()
  analyzer = DataAnalyzer(args.visualize)
  analyzer.load_data()
  # analyzer.analyze()