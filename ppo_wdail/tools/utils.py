import glob
import os

import torch
import torch.nn as nn

from ppo_wdail.tools.envs import VecNormalize
import pandas as pd
import ppo_wdail.tools.config as config
import numpy as np
import random

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

writer = []

results4train = {
    'Train reward': 0.0,
    'Train steps':0,
}

results4train_test = {
    'Train reward': 0.0,
    'Train steps': 0,
    'min reward in episodes':0,
    'max reward in episodes':0,
    'mean reward in episodes':0,
}
def recordTrainResults_test(results, time_step):
    results4train_test['Train reward'] = results[0]
    results4train_test['Train steps'] = results[1]
    results4train_test['min reward in episodes'] = results[2]
    results4train_test['max reward in episodes'] = results[3]
    results4train_test['mean reward in episodes'] = results[4]

    write2tensorboard(results=results4train_test, time_step=time_step)



results4evaluate = {
    'Evaluate mean reward': 0.0,
    'Evaluate mean steps': 0,
    'Window Evaluate reward': 0.0
}



count_model = 0

results4loss = {
    'ppo_value': 0.0,
    'ppo_loss': 0.0,
    'ppo_entropy': 0.0,
    'ppo_total_loss': 0.0
}
def recordLossResults(results, time_step):
    results4loss['ppo_value'] = results[0]
    results4loss['ppo_loss'] = results[1]
    results4loss['ppo_entropy'] = results[2]
    results4loss['ppo_total_loss'] = results[3]

    write2tensorboard(results=results4loss, time_step=time_step)


results4td3loss = {
    'learning rate decay': 0.0,
    'critic_loss': 0.0,
    'actor_loss': 0.0
}
def recordTD3LossResults(results, time_step):
    results4td3loss['learning rate decay'] = results[0]
    results4td3loss['critic_loss'] = results[1]
    results4td3loss['actor_loss'] = results[2]

    write2tensorboard(results=results4td3loss, time_step=time_step)

results4Disloss = {
    'dis_loss': 0.0,
    'dis_gp': 0.0,
    'dis_entropy': 0.0,
    'dis_total_loss': 0.0
}
def recordDisLossResults(results, time_step):
    results4Disloss['dis_loss'] = results[0]
    results4Disloss['dis_gp'] = results[1]
    results4Disloss['dis_entropy'] = results[2]
    results4Disloss['dis_total_loss'] = results[3]

    write2tensorboard(results=results4Disloss, time_step=time_step)

def recordTrainResults(results, time_step):

    results4train['Train reward'] = results[0]
    results4train['Train steps'] = results[1]

    write2tensorboard(results=results4train, time_step=time_step)


results4train_gail = {
    'Train reward': 0.0,
    'Train steps': 0,
    'Expert reward': 0.0,
}
def recordTrainResults_gail(results, time_step):
    results4train_gail['Train reward'] = results[0]
    results4train_gail['Train steps'] = results[1]
    results4train_gail['Expert reward'] = results[2]

    write2tensorboard(results=results4train_gail, time_step=time_step)



def recordEvaluateResults(results, time_step):
    results4evaluate['Evaluate mean reward'] = results[0]
    results4evaluate['Evaluate mean steps'] = results[1]
    results4evaluate['Window Evaluate reward'] = results[2]

    write2tensorboard(results=results4evaluate, time_step=time_step)

def write2tensorboard(results, time_step):
    titles = results.keys()
    for title in titles:
        writer.add_scalar(title, results[title], time_step)

def store_results(evaluations, number_of_timesteps, params, S_time=0, E_time=3600):
    """Store the results of a run.

    Args:
        evaluations:
        number_of_timesteps (int):
        loss_aggregate (str): The name of the loss aggregation used. (sum or mean)
        loss_function (str): The name of the loss function used.

    Returns:
        None
    """

    df = pd.DataFrame.from_records(evaluations)
    number_of_trajectories = len(evaluations[0]) - 1
    columns = ["reward_{}".format(i) for i in range(number_of_trajectories)]
    columns.append("timestep")
    df.columns = columns
    log_save_name = Log_save_name(params)
    # timestamp = time.time()
    timestamp = np.around((E_time-S_time)/3600,2)
    results_fname = 'FinalEvaluation_'+log_save_name+'_tsteps_{}_consume_time_{}_results.csv'\
        .format(number_of_timesteps,
                timestamp)
    df.to_csv(str(config.results_dir / results_fname))


def Save_model_dir(algo_id, env_id):
    fname = '{}_{}'.format(algo_id, env_id)
    # dir_rela = os.path.join(config.trained_model_dir_rela,fname)
    dir_abs = config.trained_model_dir/fname

    if not dir_abs.is_dir():
        dir_abs.mkdir()

    return dir_abs

def Save_trained_model(count, num, model, model_dir, save_condition, eprewmean, reward_window4Evaluate):

    # if eprewmean >= save_condition:
    if eprewmean >= save_condition and reward_window4Evaluate[-1] >= save_condition:
        count += 1
        model_fname = 'Trained_model_{}.pt'.format(count)
        path = os.path.join(model_dir, model_fname)
        # torch.save(model.state_dict(), path)
        torch.save(model.state_dict(), path)
        print('********************************************************')
        print('Save model Train mean reward {:.2f}'.format(eprewmean))
        print('Save model Evaluate mean reward {:.2f}'.format(np.mean(reward_window4Evaluate)))
        print('********************************************************')
        if count >= num:
            count = 0

    return count



def Log_save_name(params):
    save_name = params["algo"] + '_' + params["env"]["name"] + '_' + str(random.randint(0, 1000))
    return save_name

def Log_save_name4gail(params):
    save_name = params["algo"] + '_' + params["env"]["name"] + '_' + str(random.randint(0, 1000))
    return save_name

def get_train_dir(params):
    save_name = Log_save_name(params)
    train_dir = config.results_dir / save_name
    return train_dir

class Write_Result(object):
    def __init__(self, params):
        save_name = Log_save_name4gail(params)
        self.train_name = 'Train_' + save_name + '.txt'
        self.eval_name = 'Eval_' + save_name + '.txt'
        self.path = config.results_dir
        self.results4train = results4train_gail
        self.results4evaluate = results4evaluate
        self.initTrain()
        self.initEval()


    def initTrain(self):
        names = self.results4train.keys()
        with open(os.path.join(self.path, self.train_name), 'w') as Record_f:
            Record_f.write('Time_step' + '\t')
            for name in names:
                Record_f.write(name + '\t')
            Record_f.write('Other' + '\n')


    def initEval(self):
        names = self.results4evaluate.keys()
        with open(os.path.join(self.path, self.eval_name), 'w') as Record_f:
            Record_f.write('Time_step' + '\t')
            for name in names:
                Record_f.write(name + '\t')
            Record_f.write('Other' + '\n')

    def step_train(self, time_step):
        results = self.results4train
        names = results.keys()
        with open(os.path.join(self.path, self.train_name), 'a') as Record_f:
            Record_f.write(str(time_step) + '\t')
            for name in names:
                Record_f.write(str(np.around(results[name], decimals=4)) + '\t')
            Record_f.write(str(0) + '\n')

    def step_eval(self, time_step):
        results = self.results4evaluate
        names = results.keys()
        with open(os.path.join(self.path, self.train_name), 'a') as Record_f:
            Record_f.write(str(time_step) + '\t')
            for name in names:
                Record_f.write(str(results[name]) + '\t')
            Record_f.write(str(0) + '\n')

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)