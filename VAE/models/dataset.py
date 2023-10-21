from torch.utils.data import Dataset
from torchvision import transforms
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
import quaternion
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob
import os
import yaml
from swarm.system.sim import SimulationManager
from torchvision.transforms import ToPILImage

class CustomDataset(Dataset):
    def __init__(self, num_images, map_pcd, name):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../swarm/config/params.yaml")
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        self.num_images = num_images
        self.images = []
        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5,), (0.5,))])
        self.transform = transforms.Compose([ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.CenterCrop(148),
                            transforms.Resize(params["vae"]["patch_size"]),
                            transforms.ToTensor(),])
        self.camera_width = params["env"]["camera_width"]
        self.camera_height = params["env"]["camera_height"]
        self.downsampling_factor = 1
        self.target_dim = params["vae"]["image_size"]
        self.map_lower_bound = [-13.0, -10.0, 0.0]
        self.map_upper_bound = [13.0, 10.0, 1.0]
        self.map_pcd = map_pcd
        self.sim_manager = SimulationManager(params)
        self.sim_manager.create_volume_field(map_pcd, visible=True)
        self.name = name

    def save_image(self, path, data):
        # png形式で保存
        plt.imsave(path, data, cmap="gray", vmin=0.0, vmax=1.0)

    def load_image(self, path):
        # png形式で読み込み
        image = plt.imread(path)
        image = image[:,:,0]
        return image
        
    def generateImage(self, idx):
        self.image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/box/{}_{}.png".format(self.name, idx))
        if os.path.exists(self.image_dir):
            if idx/self.num_images*100 % 10 == 0:
                print("Load Image: {}%".format(int(idx/self.num_images*100)))
            image = self.load_image(self.image_dir)
            return image
        else:
            if idx/self.num_images*100 % 10 == 0:
                print("Generate Image: {}%".format(int(idx/self.num_images*100)))
            # image = self.get_local_observation(self.map_pcd)
            print("not exist: {}".format(self.image_dir))
            
            image = self.sim_manager.get_local_observation(self.random_camera_pos(), self.random_camera_attitude(quat=True))
            self.save_image(self.image_dir, image)
            return image

    def random_camera_pos(self):
        # map_lower_boundとmap_upper_boundの間でランダムな位置を設定
        camera_pos = np.random.uniform(low=self.map_lower_bound, high=self.map_upper_bound)
        return camera_pos

    def random_camera_attitude(self, quat=False):
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
    
    def __getitem__(self, idx):
        image = self.generateImage(idx)
        image = self.transform(image)
        if idx == self.num_images - 1:
            print("Destroy field")
            self.sim_manager.destroy_field(self.map_pcd)
        return image, image # 入力と出力が同じになります
    
    def __len__(self):
        return self.num_images