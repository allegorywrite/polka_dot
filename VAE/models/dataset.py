from torch.utils.data import Dataset
from torchvision import transforms
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
import os

class CustomDataset(Dataset):
    def __init__(self, num_images, map_pcd):
        self.num_images = num_images
        self.images = []
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.camera_width = 640
        self.camera_height = 360
        self.downsampling_factor = 1
        self.target_dim = (128, 128)
        self.map_lower_bound = [-13.0, -10.0, 0.0]
        self.map_upper_bound = [13.0, 10.0, 1.0]
        self.map_pcd = map_pcd

    def save_image(self, path, data):
        # png形式で保存
        plt.imsave(path, data, cmap="gray", vmin=0.0, vmax=1.0)

    def load_image(self, path):
        # png形式で読み込み
        image = plt.imread(path)
        image = image[:,:,0]
        return image
        
    def generateImage(self, idx):
        self.image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/image_{}.png".format(idx))
        if os.path.exists(self.image_dir):
            if idx/self.num_images*100 % 10 == 0:
                print("Load Image: {}%".format(int(idx/self.num_images*100)))
            image = self.load_image(self.image_dir)
            return image
        else:
            if idx/self.num_images*100 % 10 == 0:
                print("Generate Image: {}%".format(int(idx/self.num_images*100)))
            image = self.get_local_observation(self.map_pcd)
            self.save_image(self.image_dir, image)
            return image

    def random_camera_pos(self):
        # map_lower_boundとmap_upper_boundの間でランダムな位置を設定
        camera_pos = np.random.uniform(low=self.map_lower_bound, high=self.map_upper_bound)
        return camera_pos

    def random_camera_attitude(self):
        # ランダムな姿勢を生成
        # ラジアンで表現される3つのランダムな角度
        angle_1 = np.random.uniform(low=-np.pi/6, high=np.pi/6, size=2)
        angle_2 = np.random.uniform(low=-np.pi, high=np.pi, size=1)

        roll = angle_1[0]    # 回転角
        pitch = angle_2[0]   # ピッチ角
        yaw = angle_1[1]    # ヨー角

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

    def get_local_observation(self, global_map_base_pc):
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        # デプス画像が上手く表示されない場合は、visible=Trueにする
        vis.create_window(window_name='3D Viewer', width=self.camera_width, height=self.camera_height, visible=True)
        render_option = vis.get_render_option()  
        render_option.point_size = 10.0
        vis.add_geometry(global_map_base_pc)
        # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        # vis.add_geometry(origin_frame)
        view_control = vis.get_view_control()

        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.camera_width, self.camera_height, fx=386.0, fy=386.0, cx=self.camera_width/2 - 0.5, cy=self.camera_height/2 -0.5)
        rot_1 = np.eye(4)
        rot_2 = np.eye(4)
        pos = self.random_camera_pos()
        attitude = self.random_camera_attitude()
        rot_1[:3, 3] = - pos
        align_mat = np.dot(Rotation.from_euler('y', -90, degrees=True).as_matrix(), Rotation.from_euler('x', 90, degrees=True).as_matrix())
        rot_2[:3,:3] = np.dot(attitude, align_mat)
        pinhole_parameters = view_control.convert_to_pinhole_camera_parameters()
        pinhole_parameters.intrinsic = intrinsic
        pinhole_parameters.extrinsic = np.dot(rot_2, rot_1)
        view_control.convert_from_pinhole_camera_parameters(pinhole_parameters)	
        # vis.run()
        # アップデート
        depth_image = vis.capture_depth_float_buffer(do_render=True)
        depth_image = np.array(depth_image)
        depth_image_exp = np.exp(-depth_image)
        depth_image_exp_bg = np.where(depth_image_exp==1, 0, depth_image_exp)
        # バイキュービック補完によるダウンサンプリング
        downsampled_image = resize(depth_image_exp_bg, self.target_dim)
        # downsampled_image = depth_image_inverted_bg
        vis.remove_geometry(global_map_base_pc)
        vis.destroy_window()

        return downsampled_image
    
    def __getitem__(self, idx):
        image = self.generateImage(idx)
        image = self.transform(image)
        return image, image # 入力と出力が同じになります
    
    def __len__(self):
        return self.num_images