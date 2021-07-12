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
from util.metrics import naive_baseline, root_mean_squared
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
data = frankaArmSeq(standardize=True, targets=tar_type)

#%%
impu = 0.00
train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
    data, impu)


# %%
