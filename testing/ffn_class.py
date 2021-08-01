import numpy as np
import torch
import wandb
import os
import time as t

from torch.utils.data import TensorDataset, DataLoader
from util.Losses import mse
from util.dataProcess import norm, denorm
from util.dataProcess import diffToAct


optim = torch.optim
nn = torch.nn


class FrankaFFNInv(nn.Module):

    def __init__(self, input_size, output_size, given_layers, lr, save_path,
                    use_cuda_if_available: bool = True, log:bool = False, log_dir:str = ""):

        super(FrankaFFNInv, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._layers = self._buid_hidden_layers(input_size, given_layers, output_size).to(self._device)


        if save_path is None:
            self._save_path = os.getcwd() + '/experiments/Franka/saved_models/model.torch'
        else:
            self._save_path = save_path

        self._learning_rate = lr
        self._optimizer = optim.Adam(self.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

        self._log = bool(log)
        if self._log:
            self._run = wandb.init(project="feedforward_mujoco", name="ffn", dir = log_dir)


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

    def forward(self, obs, next_obs):

        h=torch.cat((obs, next_obs),1)
        for layer in self._layers:
            h = layer(h)
            # h = self._dropout(x)
        return h
        # return self._layers(h)

    def train_step(self, train_obs: np.ndarray, train_next_obs: np.ndarray, train_act_targets: np.ndarray,  batch_size: int):
                    
        dataset = TensorDataset(train_obs, train_next_obs, train_act_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        inv_mse = 0
        t0 = t.time()
        b = list(loader)[0]

        for batch_idx, (obs, next_obs, act_targets) in enumerate(loader):
            obs_batch = (obs).to(self._device)
            next_obs_batch = next_obs.to(self._device)
            # obs_valid_batch = obs_valid.to(device)
            # target_batch = (targets).to(device)
            act_target_batch = (act_targets).to(self._device)

            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            pred_act = self(obs_batch, next_obs_batch)

            loss_inv = mse(act_target_batch, pred_act)

            inv_mse += loss_inv.detach().cpu().numpy()

            loss = loss_inv

            loss.backward()

            # if self._model.c.clip_gradients:
            #     torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        # avg_fwd_rmse = np.sqrt(fwd_mse / len(list(loader)))
        avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

        return avg_inv_rmse, t.time() - t0
            
    def eval(self, obs: np.ndarray, next_obs: np.ndarray, act_targets: np.ndarray,  batch_size: int):
        
        dataset = TensorDataset(obs, next_obs, act_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        inv_mse = 0
        t0 = t.time()
        b = list(loader)[0]

        for batch_idx, (obs, next_obs, act_targets) in enumerate(loader):
            with torch.no_grad():
                
                obs_batch = (obs).to(self._device)
                next_obs_batch = next_obs.to(self._device)
                # obs_valid_batch = obs_valid.to(device)
                # target_batch = (targets).to(device)
                act_target_batch = (act_targets).to(self._device)

                # Forward Pass
                pred_act = self(obs_batch, next_obs_batch)

                loss_inv = mse(act_target_batch, pred_act)

                inv_mse += loss_inv.detach().cpu().numpy()
        avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

        return avg_inv_rmse, t.time() - t0
        

    def train(self, train_obs: np.ndarray, train_next_obs: np.ndarray, train_act_targets: np.ndarray,  train_batch_size: int,
            val_obs: np.ndarray, val_next_obs: np.ndarray, val_act_targets: np.ndarray, val_batch_size: int,
            epochs, val_interval):
            
        best_rmse = np.inf
        print(self._save_path)

        wandb.watch(self,log=all)

        for i in range(epochs):
            train_inv_rmse, time = self.train_step(train_obs, train_next_obs, train_act_targets,
                                                batch_size = train_batch_size)
            print("Training Iteration {:04d}: Inverse RMSE: {:.5f}, Took {:4f} seconds".format(
                i + 1, train_inv_rmse, time))

            if self._log:
                wandb.log({"train_inverse_rmse": train_inv_rmse,
                                "epochs": i})
            
            if val_obs is not None and val_act_targets is not None and i % val_interval == 0:
                inv_rmse, time = self.eval(val_obs, val_next_obs, val_act_targets,
                                batch_size=val_batch_size)

                print("Validation: Inverse RMSE: {:.5f}".format(inv_rmse))
                if inv_rmse<best_rmse:
                    torch.save(self.state_dict(), self._save_path)

                if self._log:
                    wandb.log({"val_inverse_rmse": inv_rmse,
                            "epochs": i})

        self._run.finish()
        print("Training Done!")

    def predict(self, obs: np.ndarray, next_obs: np.ndarray, act_targets: np.ndarray,  batch_size: int):
        
        act_mean_list = []
        dataset = TensorDataset(obs, next_obs, act_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs, next_obs, act_targets) in enumerate(loader):
            with torch.no_grad():
                
                obs_batch = (obs).to(self._device)
                next_obs_batch = next_obs.to(self._device)
                # obs_valid_batch = obs_valid.to(device)
                # target_batch = (targets).to(device)

                # act_target_batch = (act_targets).to(device)

                # Forward Pass
                pred_act = self(obs_batch, next_obs_batch)

                act_mean_list.append(pred_act.cpu())

                # loss_inv = mse(act_target_batch, pred_act)

                # inv_mse += loss_inv.detach().cpu().numpy()
        # avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

        return torch.cat(act_mean_list)

    def predict_single(self, obs: np.ndarray, next_obs: np.ndarray, prev_action: np.ndarray, data : object):
        
        #normalize 

        obs = norm(obs, data, tar_type='observations')
        next_obs = norm(next_obs, data, tar_type='observation')

        with torch.no_grad():
            obs = obs.to(self._device)
            next_obs = next_obs.to(self._device)
            pred_raw = self(obs, next_obs)

        # denormalize and compute next obs
        pred_act = diffToAct(pred_raw.cpu().detach().numpy(),prev_action,data,standardize=True, post_standardize=False)

        return pred_act
