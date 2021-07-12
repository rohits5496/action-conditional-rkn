#%%

import os
import sys
# sys.path.append('.')
os.chdir("/home/i53/student/rohit_sonker/action-conditional-rkn")


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

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

nn = torch.nn

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


#%%
class FrankaFFNInv(nn.Module):

    def __init__(self, input_size, output_size, given_layers,
                    use_cuda_if_available: bool = True):
        # self._layer_norm = layer_norm
        super(FrankaFFNInv, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._layers = self._buid_hidden_layers(input_size, given_layers, output_size)
        if save_path is None:
            self._save_path = os.getcwd() + '/experiments/Franka/saved_models/model.torch'
        else:
            self._save_path = save_path
        self._learning_rate = self._model.c.learning_rate\

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches


    def _buid_hidden_layers(self, input_size, given_layers, output_size):
        '''
        return: list of hidden layers and last hidden layer dimension
        '''
        layers = []
        # input_layer
        layers.append(nn.Linear(in_features=input_size, out_features=given_layers[0]))
        layers.append(nn.ReLU())
        #hidden layers
        for i in range(0,len(given_layers)-1):
            layers.append(nn.Linear(in_features=given_layers[i], out_features=given_layers[i+1]))
            layers.append(nn.ReLU())

        # output_layer
        layers.append(nn.Linear(in_features=given_layers[-1], out_features=output_size))

        return nn.ModuleList(layers)

    def forward(self, obs):
        h=obs
        for layer in self._layers:
            h = layer(h)

        return h

def train_step(train_obs: np.ndarray, train_next_obs: np.ndarray, train_act_targets: np.ndarray,  batch_size: int, 
                model, device, optimizer):
                
    dataset = TensorDataset(train_obs, train_next_obs, train_act_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    inv_mse = 0
    t0 = t.time()
    b = list(loader)[0]

    for batch_idx, (obs, next_obs, act_targets) in enumerate(loader):
        obs_batch = (obs).to(device)
        next_obs_batch = next_obs.to(device)
        # obs_valid_batch = obs_valid.to(device)
        # target_batch = (targets).to(device)
        act_target_batch = (act_targets).to(device)

        # Set Optimizer to Zero
        optimizer.zero_grad()

        # Forward Pass
        pred_act = model(obs_batch, next_obs_batch)

        loss_inv = mse(act_target_batch, pred_act)

        inv_mse += loss_inv.detach().cpu().numpy()

        loss = loss_inv

        loss.backward()

        # if self._model.c.clip_gradients:
        #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)

        # Backward Pass Via Optimizer
        optimizer.step()

    # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
    # avg_fwd_rmse = np.sqrt(fwd_mse / len(list(loader)))
    avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

    return avg_inv_rmse, t.time() - t0
            
def eval(obs: np.ndarray, next_obs: np.ndarray, act_targets: np.ndarray,  batch_size: int, 
        model, device):
    
    dataset = TensorDataset(obs, next_obs, act_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    inv_mse = 0
    t0 = t.time()
    b = list(loader)[0]

    for batch_idx, (obs, next_obs, act_targets) in enumerate(loader):
        with torch.no_grad():
            
            obs_batch = (obs).to(device)
            next_obs_batch = next_obs.to(device)
            # obs_valid_batch = obs_valid.to(device)
            # target_batch = (targets).to(device)
            act_target_batch = (act_targets).to(device)

            # Forward Pass
            pred_act = model(obs_batch, next_obs_batch)

            loss_inv = mse(act_target_batch, pred_act)

            inv_mse += loss_inv.detach().cpu().numpy()
    avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

    return avg_inv_rmse, t.time() - t0
        

def train(train_obs: np.ndarray, train_next_obs: np.ndarray, train_act_targets: np.ndarray,  train_batch_size: int,
          val_obs: np.ndarray, val_next_obs: np.ndarray, val_act_targets: np.ndarray, val_batch_size: int,
          model, device, optimizer, epochs, val_interval, save_path):
          
    best_rmse = np.inf

    for i in range(epochs):
        train_inv_rmse, time = train_step(train_obs, train_next_obs, train_act_targets,
                                            batch_size = train_batch_size, model, device, optimizer)
        print("Training Iteration {:04d}: Inverse RMSE: {:.5f}, Took {:4f} seconds".format(
            i + 1, train_inv_rmse, time))
        # writer.add_scalar("Loss/train_forward", train_fwd_rmse, i)
        # writer.add_scalar("Loss/train_inverse", train_inv_rmse, i)
        if val_obs is not None and val_act_targets is not None and i % val_interval == 0:
            inv_rmse = eval(val_obs, val_next_obs, val_act_targets,
                            batch_size=val_batch_size, model, device)
            print("Validation: Inverse RMSE: {:.5f}".format( inv_rmse))
            if inv_rmse<best_rmse:
                torch.save(self._model.state_dict(), self._save_path)
    #     writer.add_scalar("Loss/test_forward", fwd_rmse, i)
    #     writer.add_scalar("Loss/test_inverse", inv_rmse, i)
    #     writer.flush()
    # writer.close()
    print("Training Done!")



#%%

model = FrankaFFNInv(input_size= 28, output_size=7, given_layers=[32,32])
print(model)











#%%
# Initial niave baselines
"""Data"""
dim = 14
tar_type = 'delta'  #'delta' - if to train on differences to previous actions/ current states
                    #'next_state' - if to trian directly the current ground truth actions / next states
data = frankaArmSeq(standardize=True, targets=tar_type)
impu = 0.00
train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
    data, impu)

"""Naive Baseline - Predicting Previous Actions"""
naive_baseline(train_act[:, :-1, :], train_act[:, 1:, :], data, 'actions', steps=[1, 3, 5, 10, 20], denorma=True)
naive_baseline(test_act[:, :-1, :], test_act[:, 1:, :], data, 'actions', steps=[1, 3, 5, 10, 20], denorma=True)

# """Model Parameters"""
# latent_obs_dim = 15
# act_dim = 7

# batch_size = 1000
# epochs = 250

save_path = os.getcwd() + '/experiments/Franka/saved_models/ffn/model.torch' #original





def experiment(layers, decoder_dense, act_decoder_dense, batch_size, num_basis, control_basis, latent_obs_dim, lr, epochs, load,
               expName,
               gpu):
    '''
    joints : Give a list of joints (0-6) on which you want to train on eg: [1,4]
    lr: Learning Rate
    '''

    ##### Define Model and Train It

    ffn_model = FrankaFFNInv(input_size=28,
                            output_size=7,
                            layers = [32,32])

    ##### Train the model
    if load == False:
        train(train_obs, train_next_obs, train_act_targets,batch_size,
              test_obs, test_next_obs, test_act_targets, val_batch_size=batch_size,
              model, device, optimizer, epochs,
              val_interval=1,
              save_path)


    ##### Load best model
    ffn_model.load_state_dict(torch.load(save_path))

    ##### Test RMSE
    pred = acrkn_infer.predict(test_obs, test_act, test_obs_valid, test_targets, batch_size=500)


    if tar_type=='delta':
        pred,_ = diffToAct(pred.cpu().detach().numpy(),data.test_prev_acts,data,standardize=True)
    else:
        pred = pred.cpu().detach().numpy()

    rmse = root_mean_squared(pred, test_act.cpu().detach().cpu(), data, tar='actions', denorma=True,
                             plot=[1, 2, 3])
    print('Inverse RMSE Final', rmse)