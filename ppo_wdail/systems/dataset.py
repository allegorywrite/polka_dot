import torch
from ppo_wdail.systems.datagenerator import DataGenerator
from VAE.models.swae import SWAE
import numpy as np
import os
import pickle

class ExpertDataLoader(torch.utils.data.Dataset):
    def __init__(self, params, data_size, batch_size, device):
        self.data_generator = DataGenerator()
        self.data_generator.generate_data()
        ntrain, ntest, file_batch_num = self.data_generator.__len__()
        print("Total Train Dataset Size: ", ntrain, " Total Test Dataset Size: ", ntest, "File Batch Size: ", file_batch_num)
        self.data_size = data_size
        self.batch_size = batch_size
        self.file_batch_num = file_batch_num
        self.params = params
        self.device = device
        self.load_neighbors = None
        self.dataset_array = []

    def sample_neighbor(self, neighbors_num, name="train"):
        if self.load_neighbors == neighbors_num:
            # ランダムにデータを取得
            idx = np.random.randint(0, len(self.dataset_array))
            dataset = self.dataset_array[idx]
            state = np.array(dataset)[0:-1]
            neighbor = np.array(state[3]).flatten()
            return neighbor
        else:
            print("Dataset must be loaded")
            return None

    def sample_size(self, neighbors_num, name="train"):
        if self.load_neighbors == neighbors_num:
            return len(self.dataset_array)
        else:
            print("Dataset must be loaded")
            return None

    def load_batch(self, idx=None):
        # キャッシュが存在する場合はそれを読み込む
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx))):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)), 'rb') as f:
                dataset = pickle.load(f)
            print("dataset loaded from ", os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)))
            return dataset
        train_dataset_dict, test_dataset_dict = self.data_generator.load_data(idx)
        model = SWAE(**self.params["swae"]).to(self.device)
        model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
        model.load_state_dict(torch.load(model_load_path, map_location=self.device))
        model.eval()
        dataset = []
        # 周辺UAV数でデータを分割
        for key in train_dataset_dict.keys():
            print("agent_num: ", key, " train_dataset: ", len(train_dataset_dict[key]))
            train_dataset = train_dataset_dict[key]
            batch_x = []
            batch_y = []
            # デプス画像を潜在変数に変換
            for train_data in train_dataset:
                depth = train_data[4]
                depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)
                mu, log_var = model.encode(depth)
                z = model.reparameterize(mu, log_var)
                train_data[4] = z.cpu().detach().numpy()
                state = np.array(train_data)[0:-1]
                action = np.array(train_data)[-1]
                goal = np.array(state[1]).flatten()
                velocity = np.array(state[2]).flatten()
                neighbor = np.array(state[3]).flatten()
                depth = np.array(state[4][0]).flatten()
                combined = np.concatenate((goal, velocity, depth, neighbor))
                batch_x.append(combined)
                batch_y.append(action)
            batch_x = np.array(batch_x, dtype=np.float32)
            batch_y = np.array(batch_y, dtype=np.float32)
            # バッチサイズに分割してtorchに変換
            for i in range(0, len(batch_x), self.params["ppo"]["batch_size"]):
                batch_x_torch = torch.from_numpy(batch_x[i:i+self.params["ppo"]["batch_size"]])
                batch_y_torch = torch.from_numpy(batch_y[i:i+self.params["ppo"]["batch_size"]])
                dataset.append([batch_x_torch, batch_y_torch])
            # batch_x_torchのサイズは, 3(goal position) + 3(velocity) + 6*neighbor_num + hidden state
        print("dataset size: ", len(dataset), "x", self.params["ppo"]["batch_size"])
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_id{}.pkl'.format(idx)), 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, params, device, save_memory=False, use_preprocessed_data=False):
        self.save_memory = save_memory
        self.data_generator = DataGenerator(params=params, device=device, encode_depth=save_memory)
        # TODO: use_preprocessed_dataに対応
        if not use_preprocessed_data:
            self.data_generator.generate_data()
            ntrain, ntest, file_batch_num = self.data_generator.__len__()
            print("Total Train Dataset Size: ", ntrain, " Total Test Dataset Size: ", ntest, "File Batch Size: ", file_batch_num)
            self.file_batch_num = file_batch_num

        self.data_size = params["wdail"]["data_size"]
        self.params = params
        self.device = device
        self.load_neighbors = None
        self.trajectory = {}

        self.encoder = SWAE(**self.params["swae"]).to(self.device)
        model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
        self.encoder.load_state_dict(torch.load(model_load_path, map_location=self.device))
        self.encoder.eval()

    def load_dataset(self, neighbors_num, name="train"):
        if self.load_neighbors == neighbors_num:
            print("Dataset already loaded")
            return
        self.load_neighbors = neighbors_num
        # キャッシュが存在する場合はそれを読み込む
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache/cache_{}_nn{}.pkl'.format(name, neighbors_num))
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
            print("dataset loaded from ", file_path)
            self.trajectory = dataset
            return
        train_dataset, _ = self.data_generator.load_data(neighbors_num=neighbors_num, max_data_size=self.data_size)
        state_array = []
        action_array = []

        if not train_dataset:
            print("No data")
            return

        for i in range(len(train_dataset)):
            item = train_dataset[i]
            if self.save_memory:
                z = torch.from_numpy(item[4])
                # print("Shape of Encoded Depth: ", z.shape)
            else:
                if i % 1000 == 0:
                    print("Encoding Progress: ", i, "/", len(train_dataset))
                depth = item[4]
                depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device)
                mu, log_var = self.encoder.encode(depth)
                z = self.encoder.reparameterize(mu, log_var).squeeze(0).to("cpu").detach()
            goal = torch.from_numpy(item[1])
            velocity = torch.from_numpy(item[2])
            neighbor = torch.from_numpy(item[3]).flatten()
            action = torch.from_numpy(item[5])

            if neighbor.shape[0] > 0:
                combined = torch.cat([goal, velocity, z, neighbor], dim=0)
            else:
                combined = torch.cat([goal, velocity, z], dim=0)
            state_array.append(combined)
            action_array.append(action)

        self.trajectory["state"] = state_array
        self.trajectory["action"] = action_array
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache')):
            os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../cache'))
        with open(file_path, 'wb') as f:
            pickle.dump(self.trajectory, f)

    def __len__(self):
        if self.load_neighbors is not None:
            return len(self.trajectory["state"])
        else:
            print("Dataset must be loaded")
            return None

    def __getitem__(self, idx):
        if self.load_neighbors is not None:
            return self.trajectory["state"][idx], self.trajectory["action"][idx]
        else:
            print("Dataset must be loaded")
            return None

    def sample_neighbor(self, neighbors_num, name="train"):
        if self.load_neighbors == neighbors_num:
            idx = np.random.randint(0, len(self.trajectory["state"]))
            dataset = self.trajectory["state"][idx]
            neighbor = dataset[self.params["env"]["state_dim"]+self.encoder.latent_dim:]
            return neighbor
        else:
            print("Dataset must be loaded")
            return None