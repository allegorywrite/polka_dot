import glob
import numpy as np
import concurrent.futures
from multiprocessing import Manager, cpu_count
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
from scipy.spatial.transform import Rotation
from skimage.transform import resize
from swarm.system.sim import SimulationManager

sys.path.append(str(Path(__file__).parent.parent))

# dataset will be array of 
# [neighbor_num ∈ N, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action ∈ R^3]
class CustomDataset:
	def __init__(self):
		# if torch.cuda.is_available():
		#   self.device = torch.device('cuda')
		# else:
		#   self.device = torch.device('cpu')
		# print("Initializing CustomDataset on {}...".format(self.device))
		yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
		with open(yaml_path, 'r') as f:
			params = yaml.load(f, Loader=yaml.SafeLoader)
		self.drones_num = params["env"]["num_drones"]
		self.dim_per_drone = 14
		self.replay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/replay/agents{}_*.csv".format(self.drones_num))
		manager = Manager()
		self.train_dataset = manager.list()
		self.test_dataset = manager.list()
		self.file_count = manager.Value('i', 0)
		self.data_num_max = 100000000
		self.t_start = 0
		self.goal_horizon = params["env"]["goal_horizon"]
		self.use_open3d = True
		self.use_matplotlib = False

		self.camera_width = params["env"]["camera_width"]
		self.camera_height = params["env"]["camera_height"]
		self.downsampling_factor = 1
		self.target_dim = params["vae"]["image_size"]
		self.max_vel = params["env"]["max_vel"]

		self.sencing_radius = params["env"]["sencing_radius"]
		self.test_train_ratio = 0.8

		self.depth_data_log = [[] for _ in range(self.drones_num)]

		# self.test_array = manager.list()
		self.file_size = 0
		self.file_batch_size = 100

		self.sim_manager = SimulationManager()
	
	def visualize_matplotlib(self, replay_data, global_map_world_pc, t):
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
			snapshot_vx = replay_data[t,self.dim_per_drone*agent_id+8]
			trajectory_y = replay_data[self.t_start:,self.dim_per_drone*agent_id+2]
			snapshot_y = replay_data[t,self.dim_per_drone*agent_id+2]
			snapshot_vy = replay_data[t,self.dim_per_drone*agent_id+9]
			trajectory_z = replay_data[self.t_start:,self.dim_per_drone*agent_id+3]
			snapshot_z = replay_data[t,self.dim_per_drone*agent_id+3]
			snapshot_vz = replay_data[t,self.dim_per_drone*agent_id+10]
			ax.plot(trajectory_x, trajectory_y, trajectory_z, 
							color=colors[agent_id % len(colors)], label="drone{}".format(agent_id))
			ax.scatter(snapshot_x, snapshot_y, snapshot_z, color="black", s=10)
			ax.quiver(snapshot_x, snapshot_y, snapshot_z, snapshot_vx, snapshot_vy, snapshot_vz, color="black", length=1.0, normalize=True)
		# 点群の描画(ワールド座標系のグローバルマップ)
		global_map_world_pc_downsampled = global_map_world_pc.voxel_down_sample(voxel_size=0.4)
		point_cloud = np.asarray(global_map_world_pc_downsampled.points)
		ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], color="green", s=1)
		plt.legend()
		plt.show()

	def visualize_data(self, replay_data, global_map_world_pc, local_map_base_depth, neighbor_state_local_array, goal_local, action, t, p_i_world, q_i_world, depth_data_log):
		if(self.use_open3d):
			self.visualize_open3d_world(replay_data, t, global_map_world_pc, p_i_world, q_i_world)
		if(self.use_matplotlib):
			self.visualize_matplotlib(replay_data, global_map_world_pc, t)
		# additional visualization
		sample_lens = 10
		sample_drones = 2
		if(len(depth_data_log[0]) > sample_lens):
			fig, ax = plt.subplots(sample_lens, sample_drones)
			for i in range(sample_drones):
				ax[0,i].set_title("drone {}".format(i))
				ax[0,i].axis('off')
				for j in range(sample_lens):
					ax[j,i].imshow(depth_data_log[i][j], cmap='plasma')
			plt.show()

	def generate_dataset(self, replay_file, visualize=False):
		dataset = []
		local_map_base_depth = []
		# リプレイデータの読み込み
		df = pd.read_csv(replay_file, header=None)
		replay_data = df.to_numpy()
		# マップデータの読み込み
		basename_without_ext, ext = os.path.splitext(os.path.basename(replay_file))
		map_file = "{}/../map/{}.yaml".format(os.path.dirname(replay_file), basename_without_ext)
		map_data = yaml.load(open(map_file), Loader=yaml.FullLoader)
		# print("state_seq_of_all_drones.shape = {}".format(replay_data.shape))
		vision_file_path = "{}/../vision/{}.pcd".format(os.path.dirname(replay_file), basename_without_ext)
		global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
		self.sim_manager.create_field(global_map_world_pc)
		for t in range(0, replay_data.shape[0]-1):
			# if t % 10 == 0:
			# 	print("[dataset] file = ", basename_without_ext,  "t = {}".format(t))
			if replay_data[t,0] == 0:
				self.t_start = t + 1
				continue
			for agent_id in range(0, self.drones_num):
				# 点群の読み込み
				# vision_file_path = "{}/../vision/{}.pcd_agent{}_timestep{}.pcd".format(os.path.dirname(replay_file), basename_without_ext, agent_id, t)
				# local_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
				# State, Observation, Actionの取得
				p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, q_next = self.sim_manager.getSOA_of_world(replay_data, map_data, t, agent_id)
				# ベース座標系に変換
				neighbor_state_local_array, v_i_local, goal_local, action = self.sim_manager.transform_to_local(replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, q_next)
				normalized_neighbor_state_local_array, normalized_v_i_local, normalized_goal_local = self.clip_and_normlize(neighbor_state_local_array, v_i_local, goal_local)

				depth_data = self.sim_manager.get_local_observation(p_i_world, q_i_world)
				# self.depth_data_log[agent_id].append(depth_data)

				data = [int(normalized_neighbor_state_local_array.shape[0]/2), normalized_goal_local, normalized_v_i_local, normalized_neighbor_state_local_array, depth_data, action]
				dataset.append(data)
			if(visualize and t > 30):
				print("Visualizing Data at t = {}, agent_id = {}".format(t, agent_id))
				self.visualize_data(replay_data, global_map_world_pc, local_map_base_depth, neighbor_state_local_array, goal_local, action, t, p_i_world, q_i_world, self.depth_data_log)
		self.sim_manager.destroy_field(global_map_world_pc)
		# メモリの解放
		del replay_data
		del global_map_world_pc
		del local_map_base_depth
		del neighbor_state_local_array
		del goal_local
		del action
		del depth_data

		# [neighbor_num, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action]
		return dataset

	def generate_dict(self, dataset, id=None, shuffle=True, name=None):
		if shuffle:
			random.shuffle(dataset)
		dataset_dict = dict()

		for data in dataset:
			neighbor_num = data[0]
			if neighbor_num not in dataset_dict:
				dataset_dict[neighbor_num] = []
			dataset_dict[neighbor_num].append(data)

		loader = []
		print("keys:",dataset_dict.keys())
		for key in dataset_dict.keys():
			dataset = dataset_dict[key]
			preprocessed_data = np.array(dataset, dtype=object)
			if id is not None:
				data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/preprocessed_data/batch_{}_nn{}_id{}.npy".format(name, key, id))
			else:
				data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/preprocessed_data/batch_{}_nn{}.npy".format(name, key))
			os.makedirs(os.path.dirname(data_dir), exist_ok=True)
			np.save(data_dir, preprocessed_data)
		
		return dataset_dict

	def load_dict(self, name=None, id=None):
		dataset_dict = dict()
		if id is not None:
			for key in range(0, self.drones_num):
				data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/preprocessed_data/batch_{}_nn{}_id{}.npy".format(name, key, id))
				if not os.path.exists(data_dir):
					print("Error: {} does not exist.".format(data_dir))
					continue
				dataset_dict[key] = np.load(data_dir, allow_pickle=True)
		else:
			for key in range(0, self.drones_num):
				data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/preprocessed_data/batch_{}_nn{}_id*.npy".format(name, key))
				files = glob.glob(data_dir)
				dataset_dict[key] = []
				for file in files:
					data = np.load(file, allow_pickle=True)
					print("file = {}, data.shape = {}".format(file, data.shape))
					dataset_dict[key].extend(data)
				# if not os.path.exists(data_dir):
				# 	print("Error: {} does not exist.".format(data_dir))
				# 	continue
				# dataset_dict[key] = np.load(data_dir, allow_pickle=True)
		return dataset_dict
	
	def process_file(self, file_info):
			index, file = file_info
			dataset = self.generate_dataset(file)
			len_case = len(dataset)
			# print('files = {}, len_case = {}'.format(file, len_case))
			rand = np.random.uniform(0, 1)
			if rand <= self.test_train_ratio:
				self.train_dataset.extend(dataset)
				print("[dataset] train_dataset: ", len(self.train_dataset))
			else:
				self.test_dataset.extend(dataset)
				print("[dataset] test_dataset: ", len(self.test_dataset))
			self.file_count.value += 1
			# print("[dataset] Progress: {}/{}".format(index, self.file_size))
			print("[dataset] Progress: {}/{}".format(self.file_count.value, self.file_size))
		
	def load_data(self, load=False, save=False, visualize=False, id=None):
		if load:
			print("Loading preprocessed data...")
			dataset_dict_train = self.load_dict(name="train", id=id)
			dataset_dict_test = self.load_dict(name="test", id=id)
			return dataset_dict_train, dataset_dict_test

		print("Loading Data...")
		files = glob.glob(self.replay_dir)
		print("Size of files: ", len(files))
		self.file_size = len(files)
		if visualize:
			print("Visualizing Data...")
			self.generate_dataset(files[0], True)

		print("Generating Dataset...")

		batch_itr = 0
		for i in range(0, len(files), self.file_batch_size):
			print("Processing {}th batch...".format(batch_itr))
			manager = Manager()
			self.train_dataset = manager.list()
			self.test_dataset = manager.list()
			files_batch = files[i:i+self.file_batch_size]

			with concurrent.futures.ProcessPoolExecutor(max_workers=int(cpu_count())) as executor:
				executor.map(self.process_file, enumerate(files_batch))
			del manager
			del executor

			print(batch_itr, 'th file batch, Training Dataset Size: ',len(self.train_dataset))
			print(batch_itr, 'th file batch, Total Test Dataset Size: ',len(self.test_dataset))

			if save:
				dataset_dict_train = self.generate_dict(self.train_dataset, id=batch_itr, name="train", shuffle=True)
				dataset_dict_test = self.generate_dict(self.test_dataset, id=batch_itr, name="test", shuffle=True)
			batch_itr += 1
			del self.train_dataset
			del self.test_dataset
			del dataset_dict_train
			del dataset_dict_test

		# return dataset_dict_train, dataset_dict_test
		return None, None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--visualize", action="store_true")
	parser.add_argument("--load", action="store_true")
	parser.add_argument("--save", action="store_true")
	args = parser.parse_args()
	dataset = CustomDataset()
	dataset.load_data(args.load, args.save, args.visualize)