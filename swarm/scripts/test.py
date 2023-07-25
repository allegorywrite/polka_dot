import swarm.models.dataset import CustomDataset
import VAE.models.vanilla_vae import VanillaVAE

if __name__ == '__main__':
  drone_num = 10
  in_channels = 1
  latent_dim = 200 # 潜在変数の次元数

  # データセットの読み込み
  dataset = CustomDataset(drone_num, False)
  # train_dataset, test_datasetは
  # array of [neighbor_num ∈ N, g ∈ R^3, v ∈ R^3, neighbor_0 ~ neighbor_n ∈ R^6, depth(128×128), action ∈ R^3]
  train_dataset, test_dataset = dataset.load_data()
  # エンコーダーの読み込み
  model = VanillaVAE(in_channels = in_channels, latent_dim=latent_dim)
  model.load_state_dict(torch.load('../VAE/data/model.pth'))
  model.eval()
  # デプス画像を潜在変数に変換
  for train_data in train_dataset:
    depth = train_data[4]
    depth = depth.unsqueeze(0).to(device)
    mu, log_var = model.encode(depth)
    z = model.reparameterize(mu, log_var)
    print("latent variable: ",z)
    train_data[4] = z
    state = np.array(train_data)[0:-3]
    action = np.array(train_data)[-3:]
    print("state: ",state)
    print("action: ",action)