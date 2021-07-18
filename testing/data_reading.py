#%%
import os
import sys

os.chdir("/home/i53/student/rohit_sonker/action-conditional-rkn")

# sys.path.append('../experiments/Franka')
# sys.path.append('..')

import argparse
import torch
import numpy as np


from data.frankaData_Seq_Inv import frankaArmSeq
# from rkn.acrkn.AcRknInv import AcRknInv
# from rkn_cell.acrkn_cell import AcRKNCell
# from rkn.acrkn.InverseLearning import Learn
# from rkn.acrkn.InverseInference import Infer
from util.ConfigDict import ConfigDict
from util.metrics import naive_baseline, root_mean_squared, plot_data
from util.dataProcess import diffToAct



################ Generate Data ##################
def generate_franka_data_set(data, percentage_imputation):
    train_targets = data.train_targets
    test_targets = data.test_targets

    train_obs = data.train_obs
    test_obs = data.test_obs

    rs = np.random.RandomState(seed=42)
    train_obs_valid = rs.rand(train_targets.shape[0], train_targets.shape[1], 1) < 1 - percentage_imputation
    train_obs_valid[:, :5] = True
    print("Fraction of Valid Train Observations:",
          np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
    rs = np.random.RandomState(seed=23541)
    test_obs_valid = rs.rand(test_targets.shape[0], test_targets.shape[1], 1) < 1 - percentage_imputation
    test_obs_valid[:, :5] = True
    print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

    train_act = data.train_current_acts
    test_act = data.test_current_acts

    train_act_targets = data.train_act_targets
    test_act_targets = data.test_act_targets

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(
        train_obs_valid).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(
        train_act_targets).float(), torch.from_numpy(
        test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(
        test_obs_valid).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_act_targets).float(),



def generate_franka_data_set_ffn(data, percentage_imputation):
    # train_targets = data.train_targets
    # test_targets = data.test_targets

    # train_obs = data.train_obs
    # test_obs = data.test_obs

    # rs = np.random.RandomState(seed=42)
    # train_obs_valid = rs.rand(train_targets.shape[0], train_targets.shape[1], 1) < 1 - percentage_imputation
    # train_obs_valid[:, :5] = True
    # print("Fraction of Valid Train Observations:",
    #       np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
    # rs = np.random.RandomState(seed=23541)
    # test_obs_valid = rs.rand(test_targets.shape[0], test_targets.shape[1], 1) < 1 - percentage_imputation
    # test_obs_valid[:, :5] = True
    # print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

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


#%% Test

from data.frankaData_FFNN_Inv import boschRobot

datapath = os.getcwd() + '/data/FrankaData/rubber_acce/without/'


# Load each file. Specify Episode Length.
# Episodize
f = 'r1_s08.npz'
data_npz = np.load(datapath + f)


#%% My data

datapath = os.getcwd() + '/data/FrankaData/mujoco/'


# Load each file. Specify Episode Length.
# Episodize
f = 'd0207_td5.npz'
data_npz2 = np.load(datapath + f)
# # remove the extra torque dimensions
# tau = data_npz2['tau'][:,0:7]

# np.savez_compressed(datapath+f, q=data_npz2['q'],
#                                     qd=data_npz2['qd'],
#                                     qdd = data_npz2['qdd'],
#                                     tau= tau,
#                                     t = data_npz2['t'])


# %%

"""Data"""
dim = 14
tar_type = 'delta'  #'delta' - if to train on differences to previous actions/ current states
                    #'next_state' - if to trian directly the current ground truth actions / next states
data = frankaArmSeq(standardize=True, targets=tar_type, data_to_use = 'new')

impu = 0.00
train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
    data, impu)


# %%
#compare data

dim = 14
tar_type = 'delta'  #'delta' - if to train on differences to previous actions/ current states
                    #'next_state' - if to trian directly the current ground truth actions / next states
data = boschRobot(standardize=True, targets=tar_type, data_to_use="new")
impu = 0.00
# train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
#     data, impu)


train_obs,train_next_obs, train_act_targets, train_act, test_obs, test_next_obs, test_act_targets, test_act = generate_franka_data_set_ffn(data,impu)


data2 = boschRobot(standardize=True, targets=tar_type, data_to_use="original")
impu = 0.00
# train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
#     data, impu)
train_obs2,train_next_obs2, train_act_targets2, train_act2, test_obs2, test_next_obs2, test_act_targets2, test_act2 = generate_franka_data_set_ffn(data2,impu)

idx = 1000
plot_data(train_act[idx:idx+100], train_act2[idx:idx+100])

