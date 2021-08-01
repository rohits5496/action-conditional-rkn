#%%
import os
import time as t
import sys
# sys.path.append('.')

os.chdir("/home/i53/student/rohit_sonker/action-conditional-rkn")
log_dir = '/home/temp_store/rohit_sonker'

# os.chdir("/home/rohit/action-conditional-rkn")
# log_dir = "/home/rohit/logs"

print("Current dir : ",os.getcwd())

import argparse
import torch
import numpy as np


from data.frankaData_Seq_Inv import frankaArmSeq
from data.frankaData_FFNN_Inv import boschRobot
# from rkn.acrkn.AcRknInv import AcRknInv
# from rkn_cell.acrkn_cell import AcRKNCell
# from rkn.acrkn.InverseLearning import Learn
# from rkn.acrkn.InverseInference import Infer
from util.ConfigDict import ConfigDict
from util.metrics import naive_baseline, plot_data, root_mean_squared_simple, naive_baseline_simple, plot_data
from util.dataProcess import diffToAct

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.Losses import mse
from util.dataProcess import norm, denorm
from matplotlib import pyplot as plt
import wandb

optim = torch.optim
nn = torch.nn


from testing.ffn_class import FrankaFFNInv

def generate_franka_data_set_ffn(data, percentage_imputation):

    train_next_obs = data.train_obs[1:]
    test_next_obs = data.test_obs[1:]

    train_obs = data.train_obs[:-1]
    test_obs = data.test_obs[:-1]
    train_act = data.train_current_acts[:-1]
    test_act = data.test_current_acts[:-1]

    train_act_targets = data.train_act_targets[:-1]
    test_act_targets = data.test_act_targets[:-1]

    return (torch.from_numpy(train_obs).float(), torch.from_numpy(train_next_obs).float(), 
           torch.from_numpy(train_act_targets).float(), torch.from_numpy(train_act).float(),
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_next_obs).float(),
           torch.from_numpy(test_act_targets).float(), torch.from_numpy(test_act).float())

#%%
# Initial niave baselines
"""Data"""
dim = 14
tar_type = 'delta'  #'delta' - if to train on differences to previous actions/ current states
                    #'next_state' - if to trian directly the current ground truth actions / next states

data_to_use = 'new'
data = boschRobot(standardize=True, targets=tar_type, data_to_use=data_to_use)
impu = 0.00

train_obs,train_next_obs, train_act_targets, train_act, test_obs, test_next_obs, test_act_targets, test_act = generate_franka_data_set_ffn(data,impu)

# """Naive Baseline - Predicting Previous Actions"""

naive_baseline_simple(train_act[:-1,], train_act[1:,], data, 'actions',steps=[1,3,5,10], denorma=True )
# naive_baseline_simple(train_act[:-1,], train_act[1:,], data, 'actions',steps=[1], denorma=True, plot=1 )


if data_to_use=='original':
    save_path = os.getcwd() + '/experiments/Franka/saved_models/ffn/model.torch' 
else:
    save_path = os.getcwd() + '/experiments/Franka/saved_models/mujoco/ffn/model.torch' 

print("\n\nSave path is : ",save_path)

#%%

hidden_layers = [500,500,500]
batch_size = 1000
use_cuda_if_available = True
load = False
epochs=100

learning_rate = 4e-3

model = FrankaFFNInv(input_size=28,
                        output_size=7,
                        given_layers = hidden_layers,
                        lr=learning_rate,
                        save_path= save_path,    
                        use_cuda_if_available=True,
                        log = True)


# model.to(device)
print("Device is : ", model._device)


##### Train the model
if load == False:
    # wandb_run = wandb.init(project="acrkn", name="ffn", dir = '/home/temp_store/rohit_sonker')

    model.train(train_obs, train_next_obs, train_act_targets,batch_size,
            test_obs, test_next_obs, test_act_targets, val_batch_size=batch_size,
            epochs=epochs,
            val_interval=1)



#%%
##### Load best model
model.load_state_dict(torch.load(save_path))

##### Test RMSE
pred_raw = model.predict(test_obs, test_next_obs, test_act_targets, batch_size=batch_size)
pred = pred_raw.cpu().detach().numpy()

# pred = np.zeros([73499, 7])



if tar_type=='delta':
    prev_actions = data.test_prev_acts[:-1]
    pred,_ = diffToAct(pred,prev_actions,data,standardize=True)

#else do nothing just use pred


#%%
rmse = root_mean_squared_simple(pred, test_act.cpu().detach().cpu(), data, tar='actions', denorma=True,
                            plot=1)
print('Inverse RMSE Final', rmse)


#%%
