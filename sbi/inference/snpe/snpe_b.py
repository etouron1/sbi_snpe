# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Any, Callable, Dict, Optional, Union
from copy import deepcopy

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import Multinomial

import sbi.utils as utils
from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils import del_entries
from torch.nn.functional import kl_div
import matplotlib.pyplot as plt


class SNPE_B(PosteriorEstimator):
    def __init__(
        self,
        observation: Tensor,
        bandwith: float = 0.01,
        prop_prior: float = 0.1,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mdn",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""SNPE-B [1]. 

        [1] _Flexible statistical inference for mechanistic models of neural dynamics_,
            Lueckmann, Gonçalves et al., NeurIPS 2017, https://arxiv.org/abs/1711.01861.

        This class implements SNPE-B. SNPE-B trains across multiple rounds with a
        importance-weighted log-loss. This will make training converge directly to the true posterior.
        Thus, SNPE-B is not limited to Gaussian proposal.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during training.
        """

 
        self._observation = observation
        self._bandwith = bandwith
        self._prop_prior = prop_prior
        
        kwargs = del_entries(locals(), entries=("self", "__class__", "observation", "bandwith", "prop_prior"))
        super().__init__(**kwargs)

    def _log_prob_proposal_posterior(
        self, 
        theta: Tensor, 
        x: Tensor, 
        masks: Tensor, 
        proposal: Optional[Any],
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x: Batch of data.
            masks: Indicate if the data (theta, x) of the batch 
                are sampled from the prior or from the proposal.
            proposal: Proposal distribution.

        Returns:
            Importance-weighted log probability.
        """

        prop = 1.0/(self._round+1)
        prior = torch.exp(self._prior.log_prob(theta))
        proposal = torch.zeros(theta.size(0), device=theta.device)
        for density in self._proposal_roundwise:
            proposal += prop*torch.exp(density.log_prob(theta))
        importance_weights = prior/proposal
        #importance_weights = importance_weights/self._normalisation_cst

        return importance_weights*self._neural_net.log_prob(theta, x)
    


    def _calibration_kernel(self, x: Tensor):
        
        theta = self._prior.sample((100, ))

        posterior_obs = deepcopy(self._posterior)
        posterior_x = deepcopy(self._posterior)

        proposal_obs = posterior_obs.set_default_x(self._observation)
   
        calibration_kernel = []

        for x_item in x:
            proposal_x = posterior_x.set_default_x(x_item)
            kl_divergence = kl_div(proposal_x.log_prob(theta), proposal_obs.log_prob(theta), log_target=True)
            calibration_kernel.append(torch.exp(-kl_divergence/self._bandwith))
        return torch.tensor(calibration_kernel)


    def train(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> DensityEstimator:
        r"""Return density estimator that approximates directly the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round. Not supported for
                SNPE-A.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        kwargs = utils.del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
            ),
        )

        self._round = max(self._data_round_index)
        # if self._round > 0:
            
            
            #batch_size = len(self._theta_roundwise[0])
            # theta = torch.cat(self._theta_roundwise, dim=0)
            #self._N = theta.size(0)
            
            # x = torch.cat(self._x_roundwise, dim=0)
            # prop = 1.0/(self._round+1)
     
            # proposal = torch.zeros(theta.size(0))
            # for density in self._proposal_roundwise:
            #     proposal += prop*torch.exp(density.log_prob(theta))
        
            # if prop_ess < 30:

            #     self._resampling = True

            #     indices = Multinomial(theta.size(0)-1, importance_weights).sample()
            #     theta = torch.index_select(input = theta,dim=0, index=indices.int())
            #     x = torch.index_select(input = x,dim=0, index=indices.int())
            #     importance_weights = torch.ones(theta.size(0))/theta.size(0)

            #     self._theta_roundwise = list(torch.split(theta, batch_size))
            #     self._x_roundwise = list(torch.split(x, batch_size))

            #     self._resample.append(self._resampling)
            

        return super().train(**kwargs)
