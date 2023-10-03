import numpy as np
import torch
import utils

from tensorboardX import SummaryWriter
from collections import deque
import time
import os
import shutil

def wdail_train(params, env, agent, discriminator, rollouts, gail_train_loader, device):

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

    i_update = 0
    time_step = 0
    dis_init = True
    epinfobuf = deque(maxlen=10)
    epgailbuf = deque(maxlen=10)
    episode_rewards = deque(maxlen=10)

    # TODO: env関数の検証
    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    
    write_result = utils.Write_Result(params=params)

    while i_update < epoch_num:
        i_update += 1
        epinfos = []

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
            
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

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

        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
            % (i_update, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))
        
    