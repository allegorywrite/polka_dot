import math
import numpy as np
import quaternion
import os
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation
from skimage.transform import resize

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BasePolkadotAviary import BasePolkadotAviary

class PolkadotAviary(BasePolkadotAviary):

    # Action Type: VEL5D = [ vx vy vz |v| wz ]
    # Observation Type: VIS = [ neighbor_num ∈ N, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128) ]

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 goal_position=None,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        
        # パラメータの読み込み
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../config/params.yaml")
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        self.sencing_radius = params["env"]["sencing_radius"]
        self.goal_horizon = params["env"]["goal_horizon"]
        self.camera_width = params["env"]["camera_width"]
        self.camera_height = params["env"]["camera_height"]
        self.image_size = params["vae"]["image_size"]
        self.max_vel = params["env"]["max_vel"]
        self.goal_position = np.array(goal_position)
        vision_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/training/vision/{}.pcd".format(params["env"]["map_name"]))
        self.global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)

    ################################################################################

    def _computeReward(self):
        rewards = {}
        for i in range(0, self.NUM_DRONES):
            rewards[i] = -1
        return rewards

    ################################################################################
    
    def _computeDone(self):
        bool_val = False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done
    
    ################################################################################
    
    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _computeDroneObservation(self, state, neighbors_state):
        
        """
        Parameters
        ----------
        state : [x, y, z, q0, q1, q2, q3, r, p, y, vx, vy, vz, wx, wy, wz, ax, ay, az, a0]
        neighbors_state : ndarray of state array

        Returns
        -------
        state_processed : [|N|, gx, gy, gz, vx, vy, vz]
        neighbors_state_processed : [x0, y0, z0, vx0, vy0, vz0, ..., xN, yN, zN, vxN, vyN, vzN]
        """
        if self.goal_position is None:
            print("[ERROR] in PolkadotAviary._computeDroneObservation()")

        MAX_LIN_VEL = self.max_vel
        MAX_PITCH_ROLL = np.pi # Full range

        p_i_world = state[0:3]
        q_i_world = np.quaternion(state[3], state[4], state[5], state[6])
        v_i_world = state[10:13]
        w_i_world = state[13:16]

        R_inverse_iw = quaternion.as_rotation_matrix(q_i_world.conjugate())

        goal_i_local = np.dot(R_inverse_iw, self.goal_position - p_i_world)
        normalized_goal = goal_i_local / self.goal_horizon
        if np.linalg.norm(goal_i_local) > self.goal_horizon:
            normalized_goal = goal_i_local / np.linalg.norm(goal_i_local)

        clipped_v_i = np.clip(v_i_world, -MAX_LIN_VEL, MAX_LIN_VEL)
        normalized_v_i = clipped_v_i / MAX_LIN_VEL

        neighbor_drones_num = 0
        neighbors_state_processed = np.zeros(6*(self.NUM_DRONES-1))
        for j in range(0, self.NUM_DRONES-1):
            p_j_world = neighbors_state[j][0:3]
            neighbor_distance = np.linalg.norm(p_j_world - p_i_world)
            if neighbor_distance > self.sencing_radius:
                continue
            v_j_world = neighbors_state[j][10:13]
            w_j_world = neighbors_state[j][13:16]
            p_ij_local = np.dot(R_inverse_iw, p_j_world - p_i_world)
            v_ij_local = np.dot(R_inverse_iw, v_j_world)
            clipped_p_j = np.clip(p_ij_local, -self.sencing_radius, self.sencing_radius)
            clipped_v_j = np.clip(v_ij_local, -MAX_LIN_VEL, MAX_LIN_VEL)
            normalized_p_j = clipped_p_j / self.sencing_radius
            normalized_v_j = clipped_v_j / MAX_LIN_VEL
            # 相対速度で計算する場合
			# v_ij_local = np.dot(R_inverse_iw, v_j_world - v_i_world) + np.cross(w_i_world, p_ij_local)
            state_clipped_and_normalized = np.concatenate((normalized_p_j, normalized_v_j))
            neighbors_state_processed[6*j:6*(j+1)] = state_clipped_and_normalized
            neighbor_drones_num += 1
        
        state_processed = np.concatenate((np.array([neighbor_drones_num]), normalized_goal, normalized_v_i))
        depth = self._getDroneVision(state)
        
        return state_processed, neighbors_state_processed, depth
    
    ################################################################################

    def _getDroneVision(self, state):
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
		# デプス画像が上手く表示されない場合は、visible=Trueにする
        vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=True)
        render_option = vis.get_render_option()  
        render_option.point_size = 10.0
        vis.add_geometry(self.global_map_world_pc)
        view_control = vis.get_view_control()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.camera_width, self.camera_height, fx=386.0, fy=386.0, cx=self.camera_width/2 - 0.5, cy=self.camera_height/2 -0.5)

        # カメラの姿勢を設定
        rot_1 = np.eye(4)
        rot_2 = np.eye(4)
        pos = state[0:3]
        quat = np.quaternion(state[3], state[4], state[5], state[6])
        attitude = quaternion.as_rotation_matrix(quat)
        rot_1[:3, 3] = - pos
        align_mat = np.dot(Rotation.from_euler('y', -90, degrees=True).as_matrix(), Rotation.from_euler('x', 90, degrees=True).as_matrix())
        rot_2[:3,:3] = np.dot(attitude, align_mat)
        pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
        pinhole_parameters.intrinsic = intrinsic
        pinhole_parameters.extrinsic = np.dot(rot_2, rot_1)

        view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)	
        # vis.run()
        depth_image = vis.capture_depth_float_buffer(do_render=True)
        depth_image = np.array(depth_image)
        depth_image_exp = np.exp(-depth_image)
        depth_image_exp_bg = np.where(depth_image_exp==1, 0, depth_image_exp)
        # 最大値によるダウンサンプリング
        # downsampled_image = self.downsample_max(depth_image_exp_bg, self.downsampling_factor)
        # バイキュービック補完によるダウンサンプリング
        downsampled_image = resize(depth_image_exp_bg, self.image_size)
        vis.remove_geometry(self.global_map_world_pc)
        vis.destroy_window()

        return downsampled_image

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_v_jel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_v_jel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_v_jel_xy,
                                               clipped_v_jel_z
                                               )

        normalized_p_jos_xy = clipped_pos_xy / MAX_XY
        normalized_p_jos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_v_jel_xy = clipped_v_jel_xy / MAX_LIN_VEL_XY
        normalized_v_jel_z = clipped_v_jel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_p_jos_xy,
                                      normalized_p_jos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_v_jel_xy,
                                      normalized_v_jel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_v_jel_xy,
                                      clipped_v_jel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_v_jel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_v_jel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlockAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
