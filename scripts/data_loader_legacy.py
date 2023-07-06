import glob
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count
from torch import nn, tanh, relu
import sys
from pathlib import Path
import resource
import torch
import random 

# `systems` を含むディレクトリへの相対パス（適宜変更してください）
sys.path.append(str(Path(__file__).parent.parent))

from systems.doubleintegrator import DoubleIntegrator

class DoubleIntegratorParam():
    def __init__(self):
        self.env_name = 'DoubleIntegrator'
        self.env_case = None

        # some path param
        self.preprocessed_data_dir = 'data/preprocessed_data/'
        self.default_instance = "map_8by8_obst06_agents004_ex000235.yaml"
        self.current_model = 'il_current.pt'

        # dont change these sim param (same as ORCA baseline)
        self.n_agents = 4
        self.r_comm = 3. 
        self.r_obs_sense = 3.
        self.r_agent = 0.15 
        self.r_obstacle = 0.5
        self.v_max = 0.5
        self.a_max = 2.0 
        self.v_min = -1*self.v_max
        self.a_min = -1*self.a_max

        # sim 
        self.sim_t0 = 0
        self.sim_tf =200
        self.sim_dt = 0.5
        self.sim_times = np.arange(self.sim_t0,self.sim_tf,self.sim_dt)
        self.sim_nt = len(self.sim_times)
        self.plots_fn = 'plots.pdf'

        # for batching/speed
        self.max_neighbors = 6
        self.max_obstacles = 6
        self.rollout_batch_on = True
                
        # safety parameters
        self.safety = "cf_di_2" 
        if self.safety == "cf_di_2": # 'working di 2' parameters
            self.pi_max = 1.5 # 0.05 
            self.kp = 0.025 # 0.01 
            self.kv = 1.0 # 2.0 
            self.cbf_kp = 0.035 # 0.5
            self.cbf_kd = 0.5 # 2.0          
        self.Delta_R = 2*(0.5*0.05 + 0.5**2/(2*2.0))
        self.epsilon = 0.01

        # imitation learning param (IL)
        self.il_load_loader_on = False
        self.training_time_downsample = 50 #10
        self.il_train_model_fn = '../models/doubleintegrator/il_current.pt'
        self.il_imitate_model_fn = '../models/doubleintegrator/rl_current.pt'
        self.il_load_dataset_on = True
        self.il_test_train_ratio = 0.85
        self.il_batch_size = 4096*8
        self.il_n_epoch = 100
        self.il_lr = 1e-3
        self.il_wd = 0 #0.0002
        self.il_n_data = None # 100000 # 100000000
        self.il_log_interval = 1
        self.il_load_dataset = ['orca','centralplanner'] # 'random','ring','centralplanner'
        self.il_controller_class = 'Barrier' # 'Empty','Barrier',
        self.il_pretrain_weights_fn = None # None or path to *.tar file
        
        # dataset param
        # ex: only take 8 agent cases, stop after 10K points 
        self.datadict = dict()
        self.datadict["8"] = 10000000

        # plots
        self.vector_plot_dx = 0.25 

        # learning hyperparameters
        n,m,h,l,p = 4,2,64,16,16 # state dim, action dim, hidden layer, output phi, output rho
        self.il_phi_network_architecture = nn.ModuleList([
            nn.Linear(4,h),
            nn.Linear(h,h),
            nn.Linear(h,l)])

        self.il_phi_obs_network_architecture = nn.ModuleList([
            nn.Linear(4,h),
            nn.Linear(h,h),
            nn.Linear(h,l)])

        self.il_rho_network_architecture = nn.ModuleList([
            nn.Linear(l,h),
            nn.Linear(h,h),
            nn.Linear(h,p)])

        self.il_rho_obs_network_architecture = nn.ModuleList([
            nn.Linear(l,h),
            nn.Linear(h,h),
            nn.Linear(h,p)])

        self.il_psi_network_architecture = nn.ModuleList([
            nn.Linear(2*p+4,h),
            nn.Linear(h,h),
            nn.Linear(h,m)])

        self.il_network_activation = relu

        # plots
        self.vector_plot_dx = 0.3

class DataAnalyzer:
    def __init__(self, env, param, device, datadir):
        self.datadir = datadir
        self.train_dataset = []
        self.data_num_max = 100000
        self.env = env
        self.param = param
        self.device = device

    def make_loader(
        self,
        env,
        param,
        dataset=None,
        n_data=None,
        shuffle=False,
        batch_size=None,
        name=None):

        if dataset is None:
          raise Exception('dataset not specified')
        
        if shuffle:
          random.shuffle(dataset)

        if n_data is not None and n_data < len(dataset):
          dataset = dataset[0:n_data]

        # break by observation size
        dataset_dict = dict()
        itr = 0
        for data in dataset:
          # print("itr:{}, data shape:{}, data:{}".format(itr, data.shape, data))
          itr += 1
          num_neighbors = int(data[0])
          if env.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
            num_obstacles = int((data.shape[0] - 1 - env.state_dim_per_agent - num_neighbors*env.state_dim_per_agent - 2) / 2)
          elif env.param.env_name == 'SingleIntegratorVelSensing':
            num_obstacles = int((data.shape[0] - 1 - 2 - num_neighbors*4 - 2) / 2)

          key = (num_neighbors, num_obstacles)
          if key in dataset_dict:
            dataset_dict[key].append(data)
          else:
            dataset_dict[key] = [data]

        # Create actual batches
        loader = []
        for key, dataset_per_key in dataset_dict.items():
          num_neighbors, num_obstacles = key
          batch_x = []
          batch_y = []
          for data in dataset_per_key:
            batch_x.append(data[0:-2])
            batch_y.append(data[-2:])

          # store all the data for this nn/no-pair in a file
          batch_x = np.array(batch_x, dtype=np.float32)
          batch_y = np.array(batch_y, dtype=np.float32)
          batch_xy = np.hstack((batch_x, batch_y))

          print(name, " neighbors ", num_neighbors, " obstacles ", num_obstacles, " ex. ", batch_x.shape[0])
          print("obsrevation:", batch_x)
          print("action:", batch_y)

          with open("../{}/batch_{}_nn{}_no{}.npy".format(param.preprocessed_data_dir,name,num_neighbors,num_obstacles), "wb") as f:
            np.save(f, batch_xy, allow_pickle=False)

          # convert to torch
          batch_xy_torch = torch.from_numpy(batch_xy).float().to(self.device)

          # split data by batch size
          for idx in np.arange(0, batch_x.shape[0], batch_size):
            last_idx = min(idx + batch_size, batch_x.shape[0])
            # print("Batch of size ", last_idx - idx)
            x_data = batch_xy_torch[idx:last_idx, 0:-2]
            y_data = batch_xy_torch[idx:last_idx, -2:]
            loader.append([x_data, y_data])

        return loader

    def load_data(self):
        resource.setrlimit(resource.RLIMIT_NOFILE, (2048, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
        files = glob.glob(self.datadir)
        len_case = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
          itr = 0
          for dataset in executor.map(self.env.load_dataset_action_loss, files):
            len_case += len(dataset)
            itr += 1
            print('files = {}, len_case = {}'.format(itr, len_case))
            if len_case > self.data_num_max:
              break
            self.train_dataset.extend(dataset)
          
        print('Total Training Dataset Size: ',len(self.train_dataset))
        loader_train = self.make_loader(
        self.env,
        self.param,
        dataset=self.train_dataset,
        shuffle=True,
        batch_size=self.param.il_batch_size,
        n_data=self.param.il_n_data,
        name = "train")

        return loader_train

    def analyze(self):
        # データを分析
        print("Analyze data")

if __name__ == '__main__':

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  datapattern = "agents008"
  datadir = "../data/training/doubleintegrator/part*/central/*{}*.npy".format(datapattern)

  param = DoubleIntegratorParam()
  env = DoubleIntegrator(param)
  analyzer = DataAnalyzer(env, param, device, datadir)
  analyzer.load_data()

  # analyzer.analyze()