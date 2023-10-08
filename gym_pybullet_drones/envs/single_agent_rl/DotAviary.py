import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseDotAviary import ActionType, ObservationType, BaseDotAviary

import quaternion
import os
import open3d as o3d
from scipy.spatial.transform import Rotation
from skimage.transform import resize

class DotAviary(BaseDotAviary):
    """Single agent RL problem: hover at position."""

    # Action Type: VEL5D = [ vx vy vz |v| wz ]
    # Observation Type: VIS = [ g ∈ R^3, v ∈ R^3, depth(128×128) ]

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 params=None,
                 goal_position=None
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
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
        
        self.sencing_radius = params["env"]["sencing_radius"]
        self.goal_horizon = params["env"]["goal_horizon"]
        self.camera_width = params["env"]["camera_width"]
        self.camera_height = params["env"]["camera_height"]
        self.image_size = params["vae"]["image_size"]
        self.max_vel = params["env"]["max_vel"]
        self.goal_position = np.array(goal_position)
        vision_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/training/vision/{}.pcd".format(params["env"]["map_name"]))
        self.global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        return -1

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # TODO: DONEの条件を変更する
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _computeDroneObservation(self, state):

        """
        Parameters
        ----------
        state : [x, y, z, q0, q1, q2, q3, r, p, y, vx, vy, vz, wx, wy, wz, ax, ay, az, a0]
        neighbors_state : ndarray of state array

        Returns
        -------
        state_processed : [gx, gy, gz, vx, vy, vz]
        depth : [128, 128]
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

        state_processed = np.concatenate((normalized_goal, normalized_v_i))
        depth = self._getDroneVision(state)

        return state_processed, depth

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
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
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
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
