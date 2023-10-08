import numpy as np
import torch
import utils

from tensorboardX import SummaryWriter
from collections import deque
import time
import os
import shutil

from VAE.models.vanilla_vae import VanillaVAE

def obs_to_vector(obs, model, gail_train_loader, neighbors_num):
    state = obs["state"]
    depth = obs["depth"]
    if obs.get("neighbor", False):
        neighbor = obs["neighbor"]
    else:
        neighbor = gail_train_loader.dataset.sample_neighbor(neighbors_num)
    mu, log_var = model.encode(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0))            
    z = model.reparameterize(mu, log_var)
    obs_vector = torch.cat([state, neighbor, z], dim=0)
    return obs_vector

def wdail_train(params, env, agent, discriminator, rollouts, gail_train_dataset, device):

    log_save_name = utils.Log_save_name4gail(params)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utils.writer = SummaryWriter(log_save_path)

    # 全ステップ数
    total_steps = params["wdail"]["total_steps"]
    # 更新ステップ数
    update_steps = params["wdail"]["batch_num"]
    # エポック数
    epoch_num = np.floor(total_steps / update_steps)
    # エージェント数
    num_agents = params["env"]["num_drones"]

    i_update = 0
    time_step = 0
    dis_init = True
    epinfobuf = deque(maxlen=10)
    epgailbuf = deque(maxlen=10)
    episode_rewards = deque(maxlen=10)
    
    write_result = utils.Write_Result(params=params)

    vision_encoder = VanillaVAE(in_channels=params["vae"]["in_channels"], latent_dim=params["vae"]["latent_dim"])
    model_load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../VAE/output/model.pth')
    vision_encoder.to(device)
    vision_encoder.load_state_dict(torch.load(model_load_path, map_location=device))
    vision_encoder.eval()

    while i_update < epoch_num:
        i_update += 1
        for neighbors_num in range(num_agents):
            # TODO: サンプル数が一定数以下の場合はスキップ
            sample_size = len(gail_train_dataset)
            if sample_size < update_steps:
                print("Episode: %d,  Neighbor num: %d, Time steps: %d, Skipped. Sample size: %d" % (i_update, neighbors_num, time_step, sample_size))
                continue
            gail_train_dataset.load_dataset(neighbors_num)
            gail_train_loader = torch.utils.data.DataLoader(
                gail_train_dataset,
                batch_size=params["wdail"]["gail_batch_size"],
                shuffle=True,
                drop_last=True)

            epinfos = []
            obs = env.reset()
            # TODO: env関数の検証
            obs_vector = obs_to_vector(obs=obs, model=vision_encoder, gail_train_loader=gail_train_loader, neighbors_num=neighbors_num)
            rollouts.obs[0].copy_(obs_vector)
            rollouts.to(device)

            if params["wdail"]["use_linear_lr_decay"]:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, i_update, epoch_num,
                    params["wdail"]["lr"])
                
            # 1: agentの行動を記録
            for step in range(update_steps):
                time_step += 1
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = agent.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    
                obs, reward, done, infos = env.step(action)
                obs_vector = obs_to_vector(obs=obs, model=vision_encoder, gail_train_loader=gail_train_loader, neighbors_num=neighbors_num)

                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                    
                # TODO: マスク処理
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                
                rollouts.insert(obs_vector, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = agent.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],rollouts.masks[-1]).detach()

            # 2: Discriminatorの学習
            # 3: Discriminatorの出力(報酬)を記録
            if params["wdail"]["use_gail"]:
                gail_epoch = params["wdail"]["gail_epoch"]
                if i_update < params["wdail"]["gail_threshold"]:
                    gail_epoch = params["wdail"]["gail_pre_epoch"]

                dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
                for _ in range(gail_epoch):
                    # TODO: 引数を検証
                    dis_loss, dis_gp, dis_entropy, dis_total_loss = discriminator.update(gail_train_loader, rollouts)
                    dis_losses.append(dis_loss)
                    dis_gps.append(dis_gp)
                    dis_entropys.append(dis_entropy)
                    dis_total_losses.append(dis_total_loss)
            
                if dis_init:
                    utils.recordDisLossResults(results=(np.array(dis_losses)[0],np.array(dis_gps)[0],np.array(dis_entropys)[0],np.array(dis_total_losses)[0]),time_step=0)
                    dis_init = False

                utils.recordDisLossResults(results=(np.mean(np.array(dis_losses)),np.mean(np.array(dis_gps)),np.mean(np.array(dis_entropys)),np.mean(np.array(dis_total_losses))),time_step=time_step)

                for step in range(update_steps):
                    rollouts.rewards[step] = discriminator.predict_reward(rollouts.obs[step], rollouts.actions[step], params["wdail"]["gamma"], rollouts.masks[step])
                    if rollouts.masks[step] == 1:
                        cum_gailrewards += rollouts.rewards[step].item()
                    else:
                        epgailbuf.append(cum_gailrewards)
                        cum_gailrewards=.0

            # 4: agentの学習
            # アドバンテージ関数の計算
            rollouts.compute_returns(next_value, params["wdail"]["use_gae"], params["wdail"]["gamma"],params["wdail"]["gae_lambda"], params["use_proper_time_limits"])
            value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)


            utils.recordLossResults(results=(value_loss, action_loss, dist_entropy, total_loss),time_step=time_step)
            rollouts.after_update()
            epinfobuf.extend(epinfos)
            if not len(epinfobuf):
                continue
            eprewmean = utils.safemean([epinfo['r'] for epinfo in epinfobuf])
            eplenmean = utils.safemean([epinfo['l'] for epinfo in epinfobuf])

            utils.recordTrainResults(results=(eprewmean, eplenmean, np.mean(np.array(epgailbuf))),time_step=time_step)

            write_result.step_train(time_step)

            print("Episode: %d,  Neighbor num: %d, Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
                % (i_update, neighbors_num, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))
            
    