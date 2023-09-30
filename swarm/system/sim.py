import os
import numpy as np
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import quaternion
from swarm.system.utils import *

class SimulationManager:

    def __init__(self):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        self.camera_width = params["env"]["camera_width"]
        self.camera_height = params["env"]["camera_height"]
        self.target_dim = params["vae"]["image_size"]
        self.max_vel = params["env"]["max_vel"]
        self.sencing_radius = params["env"]["sencing_radius"]
        self.goal_horizon = params["env"]["goal_horizon"]
        self.drones_num = params["env"]["num_drones"]
        self.dim_per_drone = 14
        self.t_start = 0
        self.field_enabled = False

    def create_field(self, global_map_world_pc, visible=True):
        if self.field_enabled:
            return
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=visible)
        render_option = self.o3d_vis.get_render_option()  
        render_option.point_size = 10.0
        self.o3d_vis.add_geometry(global_map_world_pc)
        self.field_enabled = True

    def destroy_field(self, global_map_world_pc):
        if self.field_enabled:
            self.o3d_vis.remove_geometry(global_map_world_pc)
            self.o3d_vis.destroy_window()
            self.field_enabled = False

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
        rot_mat = caculate_align_mat(vec_Arr)
        mesh_arrow.translate(np.array(begin))
        mesh_arrow.rotate(rot_mat, center=begin)
        mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        mesh_sphere_begin.translate(begin)
        mesh_arrow.paint_uniform_color(color)
        mesh_sphere_begin.paint_uniform_color(color)
        return mesh_arrow, mesh_sphere_begin

    def get_local_observation(self, p_i_world, q_i_world):
        view_control = self.o3d_vis.get_view_control()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.camera_width, self.camera_height, fx=386.0, fy=386.0, cx=self.camera_width/2 - 0.5, cy=self.camera_height/2 -0.5)

        rot_1 = np.eye(4)
        rot_2 = np.eye(4)
        attitude = safe_as_rotation_matrix(q_i_world.conjugate())
        rot_1[:3, 3] = - p_i_world
        align_mat = np.dot(Rotation.from_euler('x', 90, degrees=True).as_matrix(), Rotation.from_euler('z', 90, degrees=True).as_matrix())
        rot_2[:3,:3] = np.dot(align_mat, attitude)
        pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
        pinhole_parameters.intrinsic = intrinsic
        pinhole_parameters.extrinsic = np.dot(rot_2, rot_1)

        view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)	
        # self.o3d_vis.run()
        depth_image = self.o3d_vis.capture_depth_float_buffer(do_render=True)
        depth_image = np.array(depth_image)
        depth_image_exp = np.exp(-depth_image)
        depth_image_exp_bg = np.where(depth_image_exp==1, 0, depth_image_exp)
        # 最大値によるダウンサンプリング
        # downsampled_image = self.downsample_max(depth_image_exp_bg, self.downsampling_factor)
        # バイキュービック補完によるダウンサンプリング
        downsampled_image = resize(depth_image_exp_bg, self.target_dim)

        return downsampled_image
    
    def clip_and_normlize(self, neighbor_state_local_array, v_i_local, goal_local):
        normalized_neighbor_state_local_array = np.empty(neighbor_state_local_array.shape)
        for i in range(0, int(neighbor_state_local_array.shape[0]/2)):
            p = neighbor_state_local_array[i*2]
            v = neighbor_state_local_array[i*2+1]
            cliped_p = np.clip(p, -self.sencing_radius, self.sencing_radius)
            clipec_v = np.clip(v, -self.max_vel, self.max_vel)
            normalized_p = cliped_p / self.sencing_radius
            normalized_v = clipec_v / self.max_vel
            normalized_neighbor_state_local_array[i*2] = normalized_p
            normalized_neighbor_state_local_array[i*2+1] = normalized_v
        cliped_v_i_local = np.clip(v_i_local, -self.max_vel, self.max_vel)
        normalized_v_i_local = cliped_v_i_local / self.max_vel
        normalized_goal_local = goal_local / self.goal_horizon

        return normalized_neighbor_state_local_array, normalized_v_i_local, normalized_goal_local
    
    def transform_to_local(self, replay_data, t, agent_id, p_i_world, q_i_world, v_i_world, goal_i):
        # Observation(Local座標型)
        R_inverse_iw = safe_as_rotation_matrix(q_i_world.conjugate())

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
                v_j_world = np.array(
                    replay_data[t,self.dim_per_drone*neighbor_id+8:self.dim_per_drone*neighbor_id+11])
                # Neighbor State(Local座標型)
                p_ij_local = np.dot(R_inverse_iw, p_j_world - p_i_world)
                v_ij_local = np.dot(R_inverse_iw, v_j_world)
                # 相対速度で計算する場合
                # v_ij_local = np.dot(R_inverse_iw, v_j_world - v_i_world) + np.cross(w_i_world, p_ij_local)
                neighbor_states = np.concatenate((neighbor_states, p_ij_local.reshape(1,-1), v_ij_local.reshape(1,-1)), axis=0)

        return neighbor_states, v_i_local, goal_i_local
    
    def visualize_open3d_world(self, replay_data=None, t=0, global_map_world_pc=None, p_i_world=None, q_i_world=None):
        # self.o3d_vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=True)
        render_option = self.o3d_vis.get_render_option()  
        render_option.point_size = 10.0
        render_option.line_width = 1.0

        for agent_id in range(0, self.drones_num):
            trajectory_x = replay_data[self.t_start:,self.dim_per_drone*agent_id+1]
            # print("trajectory_x = {}".format(trajectory_x))
            snapshot_x = replay_data[t,self.dim_per_drone*agent_id+1]
            snapshot_vx = replay_data[t,self.dim_per_drone*agent_id+8]
            trajectory_y = replay_data[self.t_start:,self.dim_per_drone*agent_id+2]
            snapshot_y = replay_data[t,self.dim_per_drone*agent_id+2]
            snapshot_vy = replay_data[t,self.dim_per_drone*agent_id+9]
            trajectory_z = replay_data[self.t_start:,self.dim_per_drone*agent_id+3]
            snapshot_z = replay_data[t,self.dim_per_drone*agent_id+3]
            snapshot_vz = replay_data[t,self.dim_per_drone*agent_id+10]
            # trajectoryの描画
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(np.array([trajectory_x, trajectory_y, trajectory_z]).T)
            lineset.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(len(trajectory_x)-1)]))
            lineset.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for i in range(len(trajectory_x)-1)]))
            self.o3d_vis.add_geometry(lineset)
            # snapshotの速度ベクトルの描画
            arrow, pos = self.create_arrow(begin=[snapshot_x, snapshot_y, snapshot_z], end=[snapshot_x+snapshot_vx, snapshot_y+snapshot_vy, snapshot_z+snapshot_vz], color=[0,0,0], normalized=True)
            if arrow is not None:
                self.o3d_vis.add_geometry(arrow)
                self.o3d_vis.add_geometry(pos)

        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        self.o3d_vis.add_geometry(global_map_world_pc)
        self.o3d_vis.add_geometry(origin_frame)

        downsampled_image = self.get_local_observation(p_i_world, q_i_world)
        self.o3d_vis.run()

        print("downsampled_image.shape = {}".format(downsampled_image.shape))
        # デプス画像の描画
        plt.imshow(downsampled_image, cmap='plasma')
        plt.show()
    
    # get [ State Observation Action ]
    def getSOA_of_world(self, replay_data, map_data, t=None, agent_id=None, sim=False):
        # # Observation(World座標型)
        # local_map_world_array = np.asarray(local_map_world_pc.points)
        # State(World座標型)
        t_i = replay_data[t,self.dim_per_drone*agent_id]
        goal_i = map_data["agents"][agent_id]["goal"]
        p_i_world = np.array(
            replay_data[t,self.dim_per_drone*agent_id+1:self.dim_per_drone*agent_id+4])
        # S = [ p, q, v, w ] の場合
        q_i_world = np.quaternion(
            replay_data[t,self.dim_per_drone*agent_id+4], 
            replay_data[t,self.dim_per_drone*agent_id+5], 
            replay_data[t,self.dim_per_drone*agent_id+6], 
            replay_data[t,self.dim_per_drone*agent_id+7])
        # S = [ p, v, q, w ] の場合
        # q_i_world = np.quaternion(
        # 	replay_data[t,self.dim_per_drone*agent_id+7], 
        # 	replay_data[t,self.dim_per_drone*agent_id+8], 
        # 	replay_data[t,self.dim_per_drone*agent_id+9], 
        # 	replay_data[t,self.dim_per_drone*agent_id+10])
        # S = [ p, q, v, w ] の場合
        v_i_world = np.array(
            replay_data[t,self.dim_per_drone*agent_id+8:self.dim_per_drone*agent_id+11])
        # S = [ p, v, q, w ] の場合
        # v_i_world = np.array(
        # 	replay_data[t,self.dim_per_drone*agent_id+4:self.dim_per_drone*agent_id+7])
        w_i_world = np.array(
            replay_data[t,self.dim_per_drone*agent_id+11:self.dim_per_drone*agent_id+14])
        
        if sim:
            return p_i_world, q_i_world, v_i_world, w_i_world, goal_i
        # Action(World座標型)
        p_next = np.array(
            replay_data[t+1,self.dim_per_drone*agent_id+1:self.dim_per_drone*agent_id+4])
        q_next = np.quaternion(
            replay_data[t+1,self.dim_per_drone*agent_id+4],
            replay_data[t+1,self.dim_per_drone*agent_id+5],
            replay_data[t+1,self.dim_per_drone*agent_id+6],
            replay_data[t+1,self.dim_per_drone*agent_id+7])
        # S = [ p, v, q, w ] の場合
        # q_next = np.quaternion(
        # 	replay_data[t+1,self.dim_per_drone*agent_id+7],
        # 	replay_data[t+1,self.dim_per_drone*agent_id+8],
        # 	replay_data[t+1,self.dim_per_drone*agent_id+9],
        # 	replay_data[t+1,self.dim_per_drone*agent_id+10])
        return p_i_world, q_i_world, v_i_world, w_i_world, goal_i, p_next, q_next

def step(state, action):
    # action:[v_x, v_y, v_z, |v|, w_z]

    SPEED_LIMIT = 1.0

    time = np.array([0])
    p_new = state[0] + SPEED_LIMIT*np.abs(action[3])*action[0:3]
    v_new = SPEED_LIMIT*np.abs(action[3])*action[0:3]
    # q_new = np.array([state[2].w, state[2].x, state[2].y, state[2].z])
    quaternion_iw = quaternion.from_euler_angles([0, 0, action[4]]) * state[2]
    q_new = np.array([quaternion_iw.w, quaternion_iw.x, quaternion_iw.y, quaternion_iw.z])
    
    w_new = np.array([0, 0, action[4]])
    
    new_state = np.concatenate([time, p_new, q_new, v_new, w_new])
    return new_state

def sim():
    # シミュレーションのセットアップ
    sim_manager = SimulationManager()
    # ファイルを読み込み
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    map_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/map/{}.yaml".format(params["env"]["map_name"]))
    vision_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/vision/{}.pcd".format(params["env"]["map_name"]))
    replay_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/replay/{}.csv".format(params["env"]["map_name"]))
    map_data = yaml.load(open(map_file_path), Loader=yaml.FullLoader)
    global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)
    df = pd.read_csv(replay_file_path, header=None)
    replay_data = df.to_numpy()
    # 初期状態
    simplay_data = np.empty((0, sim_manager.dim_per_drone * sim_manager.drones_num))
    # simplay_data = replay_data[0]
    simplay_data = np.vstack([simplay_data, replay_data[0]])
    # シミュレーション
    sim_manager.create_field(global_map_world_pc)
    sim_count = 10
    for sim_itr in range(sim_count):
        snapshot_data = []
        for agent_id in range(sim_manager.drones_num):
            # get observation
            p_i_world, q_i_world, v_i_world, w_i_world, goal_i = sim_manager.getSOA_of_world(simplay_data, map_data, agent_id=agent_id, t=sim_itr, sim=True)
            state = [p_i_world, v_i_world, q_i_world, w_i_world]
            neighbor_state_local_array, v_i_local, goal_local = sim_manager.transform_to_local(simplay_data, sim_itr, agent_id, p_i_world, q_i_world, v_i_world, goal_i)
            normalized_neighbor_state_local_array, normalized_v_i_local, normalized_goal_local = sim_manager.clip_and_normlize(neighbor_state_local_array, v_i_local, goal_local)
            depth_data = sim_manager.get_local_observation(p_i_world, q_i_world)
            # prediction step
            obs = {
                "state": np.concatenate((np.array([len(normalized_neighbor_state_local_array)/6]), normalized_goal_local, normalized_v_i_local)),
                "neighbors": normalized_neighbor_state_local_array,
                "depth": depth_data,
            }
            action, state = model.forward(obs, None, None)
            # dynamical step
            new_state = step(state, action.cpu().detach().numpy()[0])
            # action = np.array([0.5, 0.3, 0.2, 1, -0.1])
            # new_state = step(state, action)
            snapshot_data.extend(new_state)
        simplay_data = np.vstack([simplay_data, snapshot_data])
    sim_manager.visualize_open3d_world(replay_data=simplay_data, t=sim_itr, global_map_world_pc=global_map_world_pc, p_i_world=p_i_world, q_i_world=q_i_world)
    sim_manager.destroy_field(global_map_world_pc)

if __name__ == '__main__':
    sim()