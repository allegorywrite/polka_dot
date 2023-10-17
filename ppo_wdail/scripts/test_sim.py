from ppo_wdail.systems.sim import SimulationManager
import yaml
import os
import open3d as o3d
import numpy as np
import quaternion
import matplotlib.pyplot as plt

def random_camera_pos(map_lower_bound, map_upper_bound):
    # map_lower_boundとmap_upper_boundの間でランダムな位置を設定
    camera_pos = np.random.uniform(low=map_lower_bound, high=map_upper_bound)
    return camera_pos

def random_camera_attitude(quat=False):
    # ランダムな姿勢を生成
    # ラジアンで表現される3つのランダムな角度
    angle_1 = np.random.uniform(low=-np.pi/6, high=np.pi/6, size=2)
    angle_2 = np.random.uniform(low=-np.pi, high=np.pi, size=1)

    roll = angle_1[0]
    pitch = angle_1[1]
    yaw = angle_2[0]

    if quat:
        return quaternion.from_euler_angles([roll, pitch, yaw])
        
    # 回転行列を計算
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # 姿勢を表す3x3行列
    attitude = np.dot(Rz, np.dot(Ry, Rx))
    
    return attitude

if __name__ == "__main__":
    # パラメータの読み込み
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/params.yaml")
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    vision_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/training/vision/{}.pcd".format(params["env"]["map_name"]))
    global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)

    sim_manager = SimulationManager(params=params)
    sim_manager.create_volume_field(global_map_world_pc)
    sim_manager.viz_run()
    
    image = sim_manager.get_local_observation(random_camera_pos(2,3), random_camera_attitude(quat=True))

    # imageを描画
    plt.imshow(image)
    plt.show()