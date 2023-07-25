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
from scipy.spatial.transform import Rotation
from skimage.transform import resize

sys.path.append(str(Path(__file__).parent.parent))

# dataset will be array of 
# [neighbor_num ∈ N, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action ∈ R^3]
class CustomDataset:
	def __init__(self, drone_num, visualize=False):
		# if torch.cuda.is_available():
		#   self.device = torch.device('cuda')
		# else:
		#   self.device = torch.device('cpu')
		# print("Initializing CustomDataset on {}...".format(self.device))
		self.drones_num = drone_num
		self.dim_per_drone = 14
		self.replay_dir = "../data/training/replay/agents{}_*.csv".format(self.drones_num)
		self.train_dataset = []
		self.test_dataset = []
		self.visualize = visualize
		self.data_num_max = 100000
		self.t_start = 0
		self.goal_horizon = 8
		self.use_open3d = True
		self.use_matplotlib = False

		self.camera_width = 640
		self.camera_height = 360
		self.downsampling_factor = 1
		self.target_dim = (128, 128)

		self.sencing_radius = 3
		self.test_train_ratio = 0.9

	# get [ State Observation Action ]
	def getSOA_of_world(self, replay_data, map_data, local_map_world_pc, t, agent_id):
		# # Observation(World座標型)
		local_map_world_array = np.asarray(local_map_world_pc.points)
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
		return p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, local_map_world_array

	def pointcloud_to_depth(self, point_cloud_data):
		# point_cloud = np.asarray(point_cloud_data.points)
		# pixel_width = 1000
		# pixel_height = 1000
		# focal = 0.05
		# focal_x = focal * pixel_width
		# focal_y = focal * pixel_height
		# r_max = 8

		# projected_points = []
		# for i in range(point_cloud.shape[0]):
		# 	u = - int(focal_x * point_cloud[i][2] / point_cloud[i][0])
		# 	v = - int(focal_y * point_cloud[i][1] / point_cloud[i][0])
		# 	z = np.linalg.norm(point_cloud[i])
		# 	projected_points.append([u, v, z])
		# min_u = min(point[0] for point in projected_points)
		# min_v = min(point[1] for point in projected_points)
		# max_u = max(point[0] for point in projected_points)
		# max_v = max(point[1] for point in projected_points)

		# depth_data = np.full((max_u - min_u + 1, max_v - min_v + 1), np.inf)
		# for i in range(point_cloud.shape[0]):
		# 	u, v, z = projected_points[i]
		# 	scale = r_max / z
		# 	u_min = u - int(scale/2)
		# 	u_max = u + int(scale/2)
		# 	v_min = v - int(scale/2)
		# 	v_max = v + int(scale/2)
		# 	for u in range(u_min, u_max+1):
		# 		if u - min_u < 0 or u - min_u >= max_u - min_u + 1:
		# 			continue
		# 		for v in range(v_min, v_max+1):
		# 			if v - min_v < 0 or v - min_v >= max_v - min_v + 1:
		# 				continue
		# 			if z < depth_data[u - min_u, v - min_v]:
		# 				depth_data[u - min_u, v - min_v] = z
		depth_data = np.zeros((1,1))
		return depth_data
	
	def transform_to_local(self, replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, global_map_world_array, local_map_world_array):
		# Observation(Local座標型)
		R_inverse_iw = quaternion.as_rotation_matrix(q_i_world.conjugate())
		euler_iw = quaternion.as_euler_angles(q_i_world.conjugate())
		global_map_base_pc = o3d.geometry.PointCloud()
		global_map_base_pc.points = o3d.utility.Vector3dVector(global_map_world_array)
		global_map_base_pc.translate(-p_i_world)
		global_map_base_pc.rotate(R_inverse_iw, center=(0,0,0))
		# if t % 10 == 0:
		# 	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
		# 	size=0.6, origin=[0, 0, 0])
		# 	o3d.visualization.draw_geometries([global_map_base_pc, mesh_frame])
		local_map_world_pc = o3d.geometry.PointCloud()
		local_map_world_pc.points = o3d.utility.Vector3dVector(local_map_world_array)
		local_map_world_pc.translate(-p_i_world)
		local_map_world_pc.rotate(R_inverse_iw, center=(0,0,0))
		local_map_base_depth = self.pointcloud_to_depth(local_map_world_pc)
		local_map_base_array = np.asarray(local_map_world_pc.points)
		# Action(Local座標型)
		p_next_local = np.dot(R_inverse_iw, p_next - p_i_world)
		# State(Local座標型)
		v_i_local = np.dot(R_inverse_iw, v_i_world)
		goal_i_local = np.dot(R_inverse_iw, goal_i - p_i_world)
		if np.linalg.norm(goal_i_local) > self.goal_horizon:
			goal_i_local = goal_i_local / np.linalg.norm(goal_i_local) * self.goal_horizon
		neighbor_states = np.empty((0, 3))
		for neighbor_id in range(0, self.drones_num):
			if neighbor_id != agent_id:
				# Neighbor State(World座標型)
				p_j_world = np.array(
					replay_data[t,self.dim_per_drone*neighbor_id+1:self.dim_per_drone*neighbor_id+4])
				neighbor_distance = np.linalg.norm(p_j_world - p_i_world)
				if neighbor_distance > self.sencing_radius:
					continue
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
		return neighbor_states, v_i_local, goal_i_local, p_next_local, local_map_base_array, global_map_base_pc, local_map_base_depth

	def get_cross_prod_mat(self, pVec_Arr):
		# pVec_Arr shape (3)
		qCross_prod_mat = np.array([
				[0, -pVec_Arr[2], pVec_Arr[1]], 
				[pVec_Arr[2], 0, -pVec_Arr[0]],
				[-pVec_Arr[1], pVec_Arr[0], 0],
		])
		return qCross_prod_mat

	def caculate_align_mat(self, pVec_Arr):
		scale = np.linalg.norm(pVec_Arr)
		pVec_Arr = pVec_Arr/ scale
		# must ensure pVec_Arr is also a unit vec. 
		z_unit_Arr = np.array([0,0,1])
		z_mat = self.get_cross_prod_mat(z_unit_Arr)

		z_c_vec = np.matmul(z_mat, pVec_Arr)
		z_c_vec_mat = self.get_cross_prod_mat(z_c_vec)

		if np.dot(z_unit_Arr, pVec_Arr) == -1:
				qTrans_Mat = -np.eye(3, 3)
		elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
				qTrans_Mat = np.eye(3, 3)
		else:
				qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
																										z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

		qTrans_Mat *= scale
		return qTrans_Mat
	
	def create_arrow(self, begin=[0,0,0], end=[1,0,0], color=[0,0,0], normalized=False):
		if(np.array_equal(begin, end)):
			return None, None
		vec_Arr = np.array(end) - np.array(begin)
		vec_len = np.linalg.norm(vec_Arr)
		if(normalized and vec_len > 0):
			vec_Arr = vec_Arr / vec_len
			vec_len = 1.0

		mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
				cone_height= 0.2 * vec_len, 
				cone_radius= 0.06 * vec_len, 
				cylinder_height= 0.8 * vec_len,
				cylinder_radius=  0.04 * vec_len
				)
		mesh_arrow.compute_vertex_normals()
		rot_mat = self.caculate_align_mat(vec_Arr)
		mesh_arrow.translate(np.array(begin))
		mesh_arrow.rotate(rot_mat, center=begin)
		mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
		mesh_sphere_begin.translate(begin)
		mesh_arrow.paint_uniform_color(color)
		mesh_sphere_begin.paint_uniform_color(color)
		return mesh_arrow, mesh_sphere_begin

	def visualize_matplotlib(self, replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world):
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
			# 点群の描画(ワールド座標系のローカルマップ)
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
		ax2.scatter(local_map_base_array[:,0], local_map_base_array[:,1], local_map_base_array[:,2], 
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
		# fig3 = plt.figure()
		# ax3 = fig3.add_subplot(111)
		# print("depth_shape:",local_map_base_depth.shape)
		# ax3.imshow(local_map_base_depth, cmap="plasma")
		plt.show()

	def downsample_max(self, image, factor):
		out_shape = [dim//factor for dim in image.shape]
		result = np.empty(out_shape, dtype=image.dtype)
		for index, _ in np.ndenumerate(result):
				slices = tuple(slice(i*factor, (i+1)*factor) for i in index)
				result[index] = np.max(image[slices])
		return result

	def get_local_observation(self, global_map_base_pc):
		# Create a visualization window
		vis = o3d.visualization.Visualizer()
		vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=False)
		render_option = vis.get_render_option()  
		render_option.point_size = 10.0
		vis.add_geometry(global_map_base_pc)
		view_control = vis.get_view_control()
		intrinsic = o3d.camera.PinholeCameraIntrinsic(self.camera_width, self.camera_height, fx=386.0, fy=386.0, cx=self.camera_width/2 - 0.5, cy=self.camera_height/2 -0.5)
		rot = np.eye(4)
		rot[:3,:3] = np.dot(Rotation.from_euler('y', -90, degrees=True).as_matrix(), Rotation.from_euler('x', 90, degrees=True).as_matrix())
		pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
		pinhole_parameters.intrinsic = intrinsic
		pinhole_parameters.extrinsic = rot
		view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)	
		# vis.run()
		depth_image = vis.capture_depth_float_buffer(do_render=True)
		depth_image = np.array(depth_image)
		depth_image_exp = np.exp(-depth_image)
		depth_image_exp_bg = np.where(depth_image_exp==1, 0, depth_image_exp)
		# 最大値によるダウンサンプリング
		# downsampled_image = self.downsample_max(depth_image_exp_bg, self.downsampling_factor)
		# バイキュービック補完によるダウンサンプリング
		downsampled_image = resize(depth_image_exp_bg, self.target_dim)
		vis.remove_geometry(global_map_base_pc)
		vis.destroy_window()

		return downsampled_image
	
	def visualize_open3d(self, replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world):
		# open3dで描画
		agent_arrow, _ = self.create_arrow(begin=[0,0,0], end=p_next_local, color=[0,0,0], normalized=True)
		neighbor_arrow_list = []
		for agent_id in range(0, int(neighbor_state_local_array.shape[0]/2)):
			neighbor_pos = neighbor_state_local_array[agent_id*2] # x, y, z
			neighbor_vel = neighbor_state_local_array[agent_id*2+1] # vx, vy, vz
			neighbor_arrow, neighbor_pos = self.create_arrow(begin=neighbor_pos, end=neighbor_pos+neighbor_vel, color=[0,1,0], normalized=True)
			neighbor_arrow_list.append(neighbor_arrow)
			neighbor_arrow_list.append(neighbor_pos)
		
		# Create a visualization window
		vis = o3d.visualization.Visualizer()
		vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=True)
		render_option = vis.get_render_option()  
		render_option.point_size = 10.0

		origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
		vis.add_geometry(global_map_base_pc)
		vis.add_geometry(origin_frame)
		vis.add_geometry(agent_arrow)
		for nxt in neighbor_arrow_list:
				vis.add_geometry(nxt)

		view_control = vis.get_view_control()
		intrinsic = o3d.camera.PinholeCameraIntrinsic(self.camera_width, self.camera_height, fx=386.0, fy=386.0, cx=self.camera_width/2 - 0.5, cy=self.camera_height/2 -0.5)
		rot = np.eye(4)
		rot[:3,:3] = np.dot(Rotation.from_euler('y', -90, degrees=True).as_matrix(), Rotation.from_euler('x', 90, degrees=True).as_matrix())
		pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
		pinhole_parameters.intrinsic = intrinsic
		print("intrinsic = {}".format(pinhole_parameters.intrinsic.intrinsic_matrix))
		pinhole_parameters.extrinsic = rot
		view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)
		vis.run()
		depth_image = vis.capture_depth_float_buffer(do_render=True)
		depth_image = np.array(depth_image)
		depth_image_max = np.max(depth_image)
		depth_image_inverted = np.subtract(depth_image_max, depth_image)
		depth_image_inverted_bg = np.where(depth_image_inverted==depth_image_max, 0, depth_image_inverted)
		# 最大値によるダウンサンプリング
		# downsampled_image = self.downsample_max(depth_image_inverted_bg, self.downsampling_factor)
		# バイキュービック補完によるダウンサンプリング
		downsampled_image = resize(depth_image_inverted_bg, self.target_dim)
		print("downsampled_image.shape = {}".format(downsampled_image.shape))
		# デプス画像の描画
		plt.imshow(downsampled_image, cmap='plasma')
		plt.show()

	def visualize_data(self, replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world):
		if(self.use_open3d):
			self.visualize_open3d(replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world)
		if(self.use_matplotlib):
			self.visualize_matplotlib(replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world)

	def generate_dataset(self, replay_file, visualize=False):
		dataset = []
		# リプレイデータの読み込み
		df = pd.read_csv(replay_file, header=None)
		replay_data = df.to_numpy()
		# マップデータの読み込み
		basename_without_ext, ext = os.path.splitext(os.path.basename(replay_file))
		map_file = "{}/../map/{}.yaml".format(os.path.dirname(replay_file), basename_without_ext)
		map_data = yaml.load(open(map_file), Loader=yaml.FullLoader)
		print("state_seq_of_all_drones.shape = {}".format(replay_data.shape))
		vision_file_path = "{}/../vision/{}.pcd".format(os.path.dirname(replay_file), basename_without_ext)
		global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
		global_map_world_array = np.asarray(global_map_world_pc.points)
		for t in range(0, replay_data.shape[0]-1):
			observation_world_array = []
			if replay_data[t,0] == 0:
				self.t_start = t + 1
				continue
			for agent_id in range(0, self.drones_num):
				# 点群の読み込み
				vision_file_path = "{}/../vision/{}.pcd_agent{}_timestep{}.pcd".format(os.path.dirname(replay_file), basename_without_ext, agent_id, t)
				local_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
				# State, Observation, Actionの取得
				p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, local_map_world_array = self.getSOA_of_world(replay_data, map_data, local_map_world_pc, t, agent_id)
				observation_world_array.append(local_map_world_array)
				# ベース座標系に変換
				neighbor_state_local_array, v_i_local, goal_local, p_next_local, local_map_base_array, global_map_base_pc, local_map_base_depth = self.transform_to_local(replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, global_map_world_array, local_map_world_array)

				# TODO 未検証
				depth_data = self.get_local_observation(global_map_base_pc)
				data = [int(neighbor_state_local_array.shape[0]/2), goal_local, v_i_local, neighbor_state_local_array, depth_data, p_next_local]
				print("data = {}".format(depth_data))
				dataset.append(data)

			if(self.visualize and t > 160):
				self.visualize = False
				print("Visualizing Data at t = {}, agent_id = {}".format(t, agent_id))
				self.visualize_data(replay_data, observation_world_array, global_map_base_pc, local_map_base_array, local_map_base_depth, neighbor_state_local_array, goal_local, p_next_local, t, q_i_world)
		# データセットの作成(TODO)
		# [neighbor_num, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action]
		return dataset

	def save_data(self, dataset, shuffle=True, name=None):
		if shuffle:
			random.shuffle(dataset)
		dataset_dict = dict()

		for data in dataset:
			neighbor_num = data[0]
			if neighbor_num not in dataset_dict:
				dataset_dict[neighbor_num] = []
			dataset_dict[neighbor_num].append(data)

		loader = []
		for key in dataset_dict.keys():
			dataset = dataset_dict[key]
			preprocessed_data = np.array(dataset)
			data_dir = os.path.expanduser("~/polka_dot/data/training/preprocessed_data/batch_{}_nn{}.npy".format(name, key))
			os.makedirs(os.path.dirname(data_dir), exist_ok=True)
			np.save(data_dir, preprocessed_data)

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
				if np.random.uniform(0, 1) <= self.test_train_ratio:
					self.train_dataset.extend(dataset)
				else:
					self.test_dataset.extend(dataset)
		print('Total Training Dataset Size: ',len(self.train_dataset))
		print('Total Test Dataset Size: ',len(self.test_dataset))

		self.save_data(self.train_dataset, name="train", shuffle=True)
		self.save_data(self.test_dataset, name="test", shuffle=True)
		return self.train_dataset, self.test_dataset

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--visualize", action="store_true")
  parser.add_argument("--drone_num", type=int, default=3)
  args = parser.parse_args()
  dataset = CustomDataset(args.drone_num, args.visualize)
  dataset.load_data()