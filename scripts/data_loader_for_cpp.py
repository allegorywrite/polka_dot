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
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))

class DataAnalyzer:
	def __init__(self, drone_num, visualize=False):
		# if torch.cuda.is_available():
		#   self.device = torch.device('cuda')
		# else:
		#   self.device = torch.device('cpu')
		# print("Initializing DataAnalyzer on {}...".format(self.device))
		self.drones_num = drone_num
		self.dim_per_drone = 14
		self.replay_dir = "../data/training/replay/agents{}_*.csv".format(self.drones_num)
		self.train_dataset = []
		self.visualize = visualize
		self.data_num_max = 100000
		self.t_start = 0
		self.goal_horizon = 8

	# get [ State Observation Action ]
	def getSOA_of_world(self, replay_data, map_data, point_cloud_data, t, agent_id):
		# Observation(World座標型)
		observation_world = np.asarray(point_cloud_data.points)
		# State(World座標型)
		t_i = replay_data[t,self.dim_per_drone*agent_id]
		goal_i = map_data["agents"][agent_id]["goal"]
		p_i_world = np.array(
			replay_data[t,self.dim_per_drone*agent_id+1:self.dim_per_drone*agent_id+4])
		# S = [ p, q, v, w ] の場合
		# q_i_world = np.quaternion(
		# 	replay_data[t,self.dim_per_drone*agent_id+4], 
		# 	replay_data[t,self.dim_per_drone*agent_id+5], 
		# 	replay_data[t,self.dim_per_drone*agent_id+6], 
		# 	replay_data[t,self.dim_per_drone*agent_id+7])
		q_i_world = np.quaternion(
			replay_data[t,self.dim_per_drone*agent_id+7], 
			replay_data[t,self.dim_per_drone*agent_id+8], 
			replay_data[t,self.dim_per_drone*agent_id+9], 
			replay_data[t,self.dim_per_drone*agent_id+10])
		v_i_world = np.array(
			replay_data[t,self.dim_per_drone*agent_id+4:self.dim_per_drone*agent_id+7])
		# S = [ p, q, v, w ] の場合
		# v_i_world = np.array(
		# 	replay_data[t,self.dim_per_drone*agent_id+8:self.dim_per_drone*agent_id+11])
		w_i_world = np.array(
			replay_data[t,self.dim_per_drone*agent_id+11:self.dim_per_drone*agent_id+14])
		# Action(World座標型)
		p_next = np.array(
			replay_data[t+1,self.dim_per_drone*agent_id+1:self.dim_per_drone*agent_id+4])

		return p_i_world, q_i_world, v_i_world, w_i_world, goal_i, observation_world, p_next

	def pointcloud_to_depth(self, point_cloud_data):
		point_cloud = np.asarray(point_cloud_data.points)
		pixel_width = 1000
		pixel_height = 1000
		focal = 0.05
		focal_x = focal * pixel_width
		focal_y = focal * pixel_height
		r_max = 8

		projected_points = []
		for i in range(point_cloud.shape[0]):
			u = - int(focal_x * point_cloud[i][2] / point_cloud[i][0])
			v = - int(focal_y * point_cloud[i][1] / point_cloud[i][0])
			z = np.linalg.norm(point_cloud[i])
			projected_points.append([u, v, z])
		min_u = min(point[0] for point in projected_points)
		min_v = min(point[1] for point in projected_points)
		max_u = max(point[0] for point in projected_points)
		max_v = max(point[1] for point in projected_points)

		depth_data = np.full((max_u - min_u + 1, max_v - min_v + 1), np.inf)
		for i in range(point_cloud.shape[0]):
			u, v, z = projected_points[i]
			scale = r_max / z
			u_min = u - int(scale/2)
			u_max = u + int(scale/2)
			v_min = v - int(scale/2)
			v_max = v + int(scale/2)
			for u in range(u_min, u_max+1):
				if u - min_u < 0 or u - min_u >= max_u - min_u + 1:
					continue
				for v in range(v_min, v_max+1):
					if v - min_v < 0 or v - min_v >= max_v - min_v + 1:
						continue
					if z < depth_data[u - min_u, v - min_v]:
						depth_data[u - min_u, v - min_v] = z
		
		return depth_data
	
	def transform_to_local(self, replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, w_i_world, goal_i, observation_world, p_next):
		# Observation(Local座標型)
		R_inverse_iw = quaternion.as_rotation_matrix(q_i_world.conjugate())
		euler_iw = quaternion.as_euler_angles(q_i_world.conjugate())
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(observation_world)
		pcd.translate(-p_i_world)
		pcd.rotate(R_inverse_iw, center=(0,0,0))
		depth_data = self.pointcloud_to_depth(pcd)
		observation_local = np.asarray(pcd.points)
		# Action(Local座標型)
		p_next_local = np.dot(R_inverse_iw, p_next - p_i_world)
		# Goal State(Local座標型)
		goal_i_local = np.dot(R_inverse_iw, goal_i - p_i_world)
		if np.linalg.norm(goal_i_local) > self.goal_horizon:
			goal_i_local = goal_i_local / np.linalg.norm(goal_i_local) * self.goal_horizon
		neighbor_states = np.empty((0, 3))
		for neighbor_id in range(0, self.drones_num):
			if neighbor_id != agent_id:
				# Neighbor State(World座標型)
				p_j_world = np.array(
					replay_data[t,self.dim_per_drone*neighbor_id+1:self.dim_per_drone*neighbor_id+4])
				# S = [ p, q, v, w ] の場合
				# v_j_world = np.array(
				# 	replay_data[t,self.dim_per_drone*neighbor_id+8:self.dim_per_drone*neighbor_id+11])
				v_j_world = np.array(
					replay_data[t,self.dim_per_drone*neighbor_id+4:self.dim_per_drone*neighbor_id+7])
				# Neighbor State(Local座標型)
				p_ij_local = np.dot(R_inverse_iw, p_j_world - p_i_world)
				v_ij_local = np.dot(R_inverse_iw, v_j_world)
				# 相対速度で計算する場合
				# v_ij_local = np.dot(R_inverse_iw, v_j_world - v_i_world) + np.cross(w_i_world, p_ij_local)
				neighbor_states = np.concatenate((neighbor_states, p_ij_local.reshape(1,-1), v_ij_local.reshape(1,-1)), axis=0)

		return neighbor_states, observation_local, goal_i_local, p_next_local, depth_data

	def generate_dataset(self, replay_file, visualize=False):
		print("visualize = {}".format(visualize))
		# リプレイデータの読み込み
		df = pd.read_csv(replay_file, header=None)
		replay_data = df.to_numpy()
		# マップデータの読み込み
		basename_without_ext, ext = os.path.splitext(os.path.basename(replay_file))
		map_file = "{}/../map/{}.yaml".format(os.path.dirname(replay_file), basename_without_ext)
		map_data = yaml.load(open(map_file), Loader=yaml.FullLoader)
		print("state_seq_of_all_drones.shape = {}".format(replay_data.shape))
		for t in range(0, replay_data.shape[0]-1):
			observation_world_array = []
			if replay_data[t,0] == 0:
				self.t_start = t + 1
				continue
			for agent_id in range(0, self.drones_num):
				# 点群の読み込み
				vision_file_path = "{}/../vision/{}.pcd_agent{}_timestep{}.pcd".format(os.path.dirname(replay_file), basename_without_ext, agent_id, t)
				point_cloud_data = o3d.io.read_point_cloud(vision_file_path)
				# State, Observation, Actionの取得
				p_i_world, q_i_world, v_i_world, w_i_world, goal_i, observation_world, p_next = self.getSOA_of_world(replay_data, map_data, point_cloud_data, t, agent_id)
				observation_world_array.append(observation_world)
				# ローカル座標系に変換
				neighbor_state_local_array, observation_local, goal_local, p_next_local, depth_data = self.transform_to_local(replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, w_i_world, goal_i, observation_world, p_next)

			if(self.visualize and t > 130):
				self.visualize = False
				print("Visualizing Data at t = {}, agent_id = {}".format(t, agent_id))
				self.visualize_data(replay_data, observation_world_array, neighbor_state_local_array, observation_local, goal_local, p_next_local, t, q_i_world, depth_data)
		# データセットの作成(TODO)
		dataset = []
		return dataset

	def load_data(self):
		print("Loading Data...")
		files = glob.glob(self.replay_dir)
		print("Size of files: ", len(files))
		len_case = 0
		if self.visualize:
			print("Visualizing Data...")
			self.generate_dataset(files[0], self.visualize)
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

	def visualize_data(self, replay_data, observation_world_array, neighbor_state_local_array, observation_local, goal_local, p_next_local, t, q_i_world, depth_data):
		# ワールド座標系
		fig = plt.figure()
		ax = fig.add_subplot(111, projection="3d")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		ax.set_xlim(-15, 15)
		ax.set_ylim(-15, 15)
		ax.set_zlim(0, 15)
		ax.set_title("Episode on global map")
		colors = ["r", "g", "b", "c", "m", "y", "k"]
		for agent_id in range(0, self.drones_num):
			# エージェントの軌跡の描画
			trajectory_x = replay_data[self.t_start:,self.dim_per_drone*agent_id+1]
			snapshot_x = replay_data[t,self.dim_per_drone*agent_id+1]
			snapshot_vx = replay_data[t,self.dim_per_drone*agent_id+4]
			trajectory_y = replay_data[self.t_start:,self.dim_per_drone*agent_id+2]
			snapshot_y = replay_data[t,self.dim_per_drone*agent_id+2]
			snapshot_vy = replay_data[t,self.dim_per_drone*agent_id+5]
			trajectory_z = replay_data[self.t_start:,self.dim_per_drone*agent_id+3]
			snapshot_z = replay_data[t,self.dim_per_drone*agent_id+3]
			snapshot_vz = replay_data[t,self.dim_per_drone*agent_id+6]
			ax.plot(trajectory_x, trajectory_y, trajectory_z, 
							color=colors[agent_id % len(colors)], label="drone{}".format(agent_id))
			ax.scatter(snapshot_x, snapshot_y, snapshot_z, color="black", s=10)
			ax.quiver(snapshot_x, snapshot_y, snapshot_z, snapshot_vx, snapshot_vy, snapshot_vz, color="black", length=1.0, normalize=True)
			# 点群の描画
			point_cloud = np.array(observation_world_array[agent_id])
			ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], color=colors[agent_id % len(colors)], s=1)
		plt.legend()
    # ローカル座標系(t=0)
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111, projection="3d")
		ax2.set_xlabel("x")
		ax2.set_ylabel("y")
		ax2.set_zlabel("z")
		ax2.set_zlim(-1, 8)
		ax2.scatter(0, 0, 0, color="black", s=10)
		ax2.set_title("Snapshot on local map (Dataset) at t = {}".format(t))
		for agent_id in range(0, int(neighbor_state_local_array.shape[0]/2)):
			neighbor_pos = neighbor_state_local_array[agent_id*2] # x, y, z
			neighbor_vel = neighbor_state_local_array[agent_id*2+1] # vx, vy, vz
			ax2.scatter(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2], 
									color=colors[agent_id % len(colors)], s=10)
			ax2.quiver(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2], 
									neighbor_vel[0], neighbor_vel[1], neighbor_vel[2], 
									color=colors[agent_id % len(colors)], length=1.0, normalize=True)
		ax2.scatter(observation_local[:,0], observation_local[:,1], observation_local[:,2], 
									color=colors[agent_id % len(colors)], s=1)
		ax2.scatter(goal_local[0], goal_local[1], goal_local[2], color="black", s=20)
		# Orientationの描画
		# orientation = np.array([1, 0, 0])
		# R_iw = quaternion.as_rotation_matrix(q_i_world)
		# orientation = np.dot(R_iw, orientation)
		# ax2.quiver(0, 0, 0, orientation[0], orientation[1], orientation[2], 
		# 							color="red", length=1.0, normalize=True)
		# Actionの描画
		ax2.quiver(0, 0, 0, p_next_local[0], p_next_local[1], p_next_local[2], 
									color="black", length=1.0, normalize=True)
		# Depthの描画
		fig3 = plt.figure()
		ax3 = fig3.add_subplot(111)
		print("depth_shape:",depth_data.shape)
		ax3.imshow(depth_data, cmap="plasma")
		plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--visualize", action="store_true")
  parser.add_argument("--drone_num", type=int, default=3)
  args = parser.parse_args()
  analyzer = DataAnalyzer(args.drone_num, args.visualize)
  analyzer.load_data()
  # analyzer.analyze()