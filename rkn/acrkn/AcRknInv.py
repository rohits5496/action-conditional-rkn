import torch
import numpy as np
from util.TimeDistributed import TimeDistributed
from rkn.acrkn.Encoder import Encoder
from rkn.acrkn.Decoder import SplitDiagGaussianDecoder, SimpleDecoder
from rkn.acrkn.AcRKNLayerInv import AcRKNLayerInv
from util.ConfigDict import ConfigDict
from typing import Tuple
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('/home/temp_store/rohit_sonker/Logs/ALRhub/fwd_noinit5')

optim = torch.optim
nn = torch.nn

def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)

def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)

class AcRknInv(nn.Module):
    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[],
            control_net_hidden_units=[60],
            trans_net_hidden_activation="Tanh",
            control_net_hidden_activation='ReLU',
            learn_trans_covar=True,
            trans_covar=0.1,
            learn_initial_state_covar=True,
            initial_state_covar=10,
            learning_rate=7e-3,
            enc_out_norm='post',
            clip_gradients=True,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, target_dim: int, lod: int, lad: int, cell_config: ConfigDict=None, use_cuda_if_available: bool = True):

        """
        :param target_dim:
        :param lod:
        :param cell_config:
        :param use_cuda_if_available:
        """
        super(AcRknInv, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._lad = lad
        self._lod = lod
        self._lsd = 2 * self._lod
        if cell_config == None:
            self.c = self.get_default_config()
        else:
            self.c = cell_config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm
        self._learning_rate = self.c.learning_rate
        # main model

        # Its not ugly, its pythonic :)
        #Build Obs Encoder
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        #Build Target Encoder
        Encoder._build_hidden_layers = self._build_target_enc_hidden_layers
        tar_enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._tar_enc = TimeDistributed(tar_enc, num_outputs=2).to(self._device)

        #Build rkn layer
        self._rkn_layer = AcRKNLayerInv(latent_obs_dim=lod, act_dim=lad, cell_config=cell_config).to(self._device)

        #Build action decoder
        SimpleDecoder._build_hidden_layers_mean = self._build_act_dec_hidden_layers_mean
        self._act_dec = TimeDistributed(SimpleDecoder(out_dim=self._lad), num_outputs=1).to(self._device)

        #Build Observation Decoder
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(out_dim=target_dim), num_outputs=2).to(self._device)

        # build (default) initial state

        if self.c.learn_initial_state_covar:
            init_state_covar = elup1_inv(self.c.initial_state_covar)
            self._init_state_covar_ul = \
                nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd), init_state_covar))
        else:
            self._init_state_covar_ul = self.c.initial_state_covar * torch.ones(1, self._lsd)

        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._init_state_covar_ul[:, :self._lod].to(self._device))
        self._icl = torch.nn.Parameter(self._init_state_covar_ul[:, self._lod:].to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)



        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_target_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_act_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_act_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, obs_valid_batch: torch.Tensor, target_batch: torch.Tensor) -> Tuple[float, float, float, float]:
        """Forward Pass oF acRKN
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :param target_batch: batch to targets as inputs for action decoder / ground truth for forward model
        :return: mean and variance
        """

        w, w_var = self._enc(obs_batch)
        w_tar, _ = self._tar_enc(target_batch)
        post_mean, post_cov, next_prior_mean, next_prior_cov  = self._rkn_layer(w, w_var, act_batch, self._initial_mean, [self._icu, self._icl, self._ics], obs_valid_batch)

        # decode current posterior for predicting actions
        act_mean = self._act_dec(torch.cat((post_mean, w_tar), dim=-1)) #Note decoded posterior is the output of update
                                                                        #step in the acrkn cell, which does not see
                                                                        #the ground truth actions while updating the posterior beliefs.
                                                                        #Ground truth actions are seen only in the ac_predict stage. One could
                                                                        #split these kalman ops, but for training it doesnt matter.

        # decode next prior for predicting the next state
        out_mean, out_var = self._dec(next_prior_mean, torch.cat(next_prior_cov, dim=-1))

        return act_mean, out_mean, out_var, w_tar

