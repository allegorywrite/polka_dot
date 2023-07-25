from models.dataset import CustomDataset
# from models.vae import Encoder, Decoder, Model
from models.vanilla_vae import VanillaVAE
from torchvision import transforms
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import optim, nn
import os

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    image_num_train = 80000
    image_num_test = 100
    batch_size = 100 # バッチサイズ
    hidden_dims = [32, 64, 128, 256, 512] # 隠れ層の次元数
    latent_dim = 200 # 潜在変数の次元数
    in_channels = 1 # 入力画像のチャンネル数(デプス画像は1)
    lr = 0.005 # 学習率
    scheduler_gamma = 0.95 # 学習率の減衰率
    weight_decay = 0.0 # 重み減衰
    kld_weight = 0.00025 # KLDの重み
    manual_seed: 1265 # 乱数シード
    epochs = 50 # エポック数
    average_loss_array = [] # 平均損失の記録用配列

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/output.png")

    vision_file_path = "map/map.pcd"
    global_map_world_pc = o3d.io.read_point_cloud(vision_file_path)

    train_dataset = CustomDataset(num_images=image_num_train, map_pcd=global_map_world_pc)
    test_dataset = CustomDataset(num_images=image_num_test, map_pcd=global_map_world_pc)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # depth_data = train_dataset.get_local_observation(global_map_world_pc)
    # plt.imshow(depth_data, cmap="plasma")
    # plt.show()

    model = VanillaVAE(in_channels = in_channels, latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = scheduler_gamma)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            real_img, labels = batch
            real_img, labels = real_img.to(device), labels.to(device)
            optimizer.zero_grad()

            results = model.forward(real_img, labels = labels)
            train_losses = model.loss_function(*results,
                M_N = kld_weight,
                batch_idx = batch_idx)
            train_loss = train_losses["loss"]
            train_loss.backward()
            optimizer.step()
            overall_loss += train_loss.item()
        scheduler.step()

        average_loss_array.append(overall_loss / (batch_idx*batch_size))

        print("\tEpoch", epoch + 1, "complete!", "\tData Size: ", batch_idx*batch_size, "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    print("Finish!!")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/loss.png")

    # 損失を対数グラフで保存してclear
    plt.plot(average_loss_array)
    plt.xlabel("Epoch")
    plt.xlim(0, epochs)
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.savefig(output_dir)
    plt.clf()

    # model.eval()
    # test_loss = 0
    # with torch.no_grad():
    #     for batch_id, batch in enumerate(test_loader):
    #         real_img, labels = batch
    #         self.curr_device = real_img.device
    #         results = self.forward(real_img, labels = labels)
    #         test_loss += self.model.loss_function(*results,
    #             M_N = kld_weight,
    #             optimizer_idx=optimizer_idx,
    #             batch_idx = batch_idx)

    print("Start testing VAE...")
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            real_img, labels = batch
            real_img, labels = real_img.to(device), labels.to(device)
            
            results = model.forward(real_img, labels = labels)
            loss = model.loss_function(*results, M_N = kld_weight)
            test_loss += loss["loss"]

    average_test_loss = test_loss / len(test_loader.dataset)
    print("Average test loss: ", average_test_loss)

    n_samples = 5
    fig, ax = plt.subplots(n_samples, 2)
    with torch.no_grad():
        for i in range(n_samples):
            test_img, labels = test_dataset[i]
            test_img = test_img.unsqueeze(0).to(device)

            labels = labels.unsqueeze(0).to(device)
            recons_img = model.generate(test_img)

            ax[i, 0].imshow(test_img.cpu().squeeze().numpy(), cmap='gray')
            ax[i, 0].set_title('Original Image')
            ax[i, 0].axis('off')

            ax[i, 1].imshow(recons_img.cpu().squeeze().numpy(), cmap='gray')
            ax[i, 1].set_title('Reconstructed Image')
            ax[i, 1].axis('off')

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/reconstructed_images.png")
    plt.savefig(output_dir)  # Save the figure before showing it