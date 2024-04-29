# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

import time
from typing import Any, Callable, Dict, Optional, Union
from copy import deepcopy

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.distributions import Multinomial, MultivariateNormal

import sbi.utils as utils
from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils import del_entries
from torch.nn.functional import kl_div
import ray
import seaborn as sns
import matplotlib.pyplot as plt




class SNPE_D(PosteriorEstimator):
    def __init__(
        self,
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

        # Catch invalid inputs.
        
        # if not ((density_estimator == "mdn") or callable(density_estimator)):
        #     raise TypeError(
        #         "The `density_estimator` passed to SNPE_B needs to be a "
        #         "callable or the string 'mdn'!"
        #     )
        # self.nets = []
        # self._resampling =False
        # self._resample = []
        #self._ess = {}
        #self._entropy = {}
        # self._observation = observation
        # self._bandwith = bandwith
        # self._prop_prior = prop_prior
        #self._var = []
        
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    # def _loss(
    #     self,
    #     theta: Tensor,
    #     x: Tensor,
    #     masks: Tensor,
    #     proposal: Optional[Any],
    #     calibration_kernel: Callable,
    #     force_first_round_loss: bool = False,
    # ) -> Tensor:
    #     """
    #     Return loss.

    #     Args:
    #         theta: Batch of parameters θ.
    #         x: Batch of data.
    #         masks: Indicate if the data (theta, x) of the batch 
    #             are sampled from the prior or from the proposal.
    #         proposal: Proposal distribution.

    #     Returns:
    #         Importance-weighted log probability.
    #     """
    #     from torch.autograd import grad
    #     from torch.autograd.functional import hessian
    #     from torch.autograd.functional import vhp
    #     from torch.distributions.multivariate_normal import MultivariateNormal
        
        
    #     with torch.enable_grad():
    #         # new_theta = torch.tensor([[1.0,2.0],[2.0,4.0], [3.0,6.0]], requires_grad=True)
            
    #         # def logq(theta):
                
    #         #     return torch.sum(torch.log(torch.sum(theta, dim=1)), dim=0)
    #         # print(torch.log(torch.sum(new_theta, dim=1)))
    #         # print(logq(new_theta))
    #         new_theta = theta.requires_grad_(True)
    #         prop = 1.0/(self._round+1)
    #         last_proposal = torch.zeros(theta.size(0))
    #         for density in self._proposal_roundwise:
    #             last_proposal += prop*torch.exp(density.log_prob(theta))
    #         def logq(theta):
    #             return torch.sum(self._neural_net.log_prob(theta, x) + torch.log(last_proposal) - self._prior.log_prob(theta))
            
    #         d_log_q_d_theta = grad(logq(new_theta), new_theta, create_graph=True)
    #         # print("d" , d_log_q_d_theta)
            
    #         first_deriv = torch.sum(d_log_q_d_theta[0]**2, dim=1)
    #         #print(first_deriv)
    #         second_deriv = torch.zeros(len(d_log_q_d_theta[0]))
    #         for i in range (theta.size(1)):
                
    #             second_deriv += grad(torch.sum(d_log_q_d_theta[0], dim=0)[i], new_theta, create_graph=True)[0][:,i]
            
    #         #print("s", second_deriv)
            
    #         # for i in range(len(new_theta)):
    #         #     def logq_sansum(theta):
    #         #         return self._neural_net.log_prob(theta, x[i]) + torch.log(last_proposal[i]) - self._prior.log_prob(theta)
    #         #     trace = 0
    #         #     for _ in range (1000):
    #         #         v = MultivariateNormal(torch.zeros(theta.size(1)), torch.eye(theta.size(1))).sample()
    #         #         Hv = vhp(logq_sansum, new_theta[i], v)
                    
    #         #         vtHv = torch.dot(v, Hv[1])
    #         #         trace +=vtHv
    #         #     trace /= 1000
    #         #     print("t", trace)

    #         # total = hessian(logq, new_theta,  create_graph=True)
    #         # diag_terms = torch.diagonal(total, dim1=1, dim2=3)
    #         # second_deriv = torch.sum(torch.sum(diag_terms, dim=0), dim=1)

    #         # result= []
    #         # for i in range (len(theta)):
    #         #     t_ = theta[i].detach().requires_grad_(True)
                
    #         #     def log_q(t):
    #         #         return self._neural_net.log_prob(t, x[i]) + proposal.log_prob(t) - self._prior.log_prob(t)
    #         #     # log_q(t_).backward()
    #         #     # d_log_q_d_theta = t_.grad
    #         #     d_log_q_d_theta = grad(log_q(t_), t_, create_graph=True)
    #         #     #print("d", d_log_q_d_theta)

    #         #     #first_deriv = torch.sum(d_log_q_d_theta[0]**2)
                
    #         #     #print()
    #         #     second_deriv = []
    #         #     #dd_log_q_dd_theta = torch.zeros(theta[i].size(0), theta[i].size(0))
    #         #     second_deriv = 0
    #         #     for j in range(len(d_log_q_d_theta[0])):
            
    #         #         #dd_log_q_dd_theta[j] = grad(d_log_q_d_theta[0][j], t_, create_graph=True)[0]
    #         #         second_deriv += grad(d_log_q_d_theta[0][j], t_, create_graph=True)[0][j]
    #         #         #second_deriv.append(dd_log_q_dd_theta)
    #         #     #print("dd", dd_log_q_dd_theta)
    #         #     #print("h", hessian(log_q, t_,  create_graph=True))
    #         #     #second_deriv = torch.trace(hessian(log_q, t_,  create_graph=True))
    #         #print("s", second_deriv)
    #         #     #term = second_deriv + 0.5 * first_deriv
    #         #     #result.append((second_deriv + 0.5 * first_deriv).item()) 
        
        
    #     #return torch.tensor(result, requires_grad=True)
        
    #     return second_deriv+0.5*first_deriv
    
    def _log_prob_proposal_posterior(
        self, 
        theta: Tensor, 
        x: Tensor, 
        masks: Tensor, 
        proposal: Optional[Any],
    ) -> Tensor:
        from torch.autograd import grad
        from torch.autograd.functional import hessian
        from torch.autograd.functional import vhp
        from torch.distributions.multivariate_normal import MultivariateNormal
        
        
        with torch.enable_grad():
 
            new_theta = theta.requires_grad_(True)
            prop = 1.0/(self._round+1)
            last_proposal = torch.zeros(theta.size(0))
            for density in self._proposal_roundwise:
                last_proposal += prop*torch.exp(density.log_prob(theta))
            def logq(theta):
                return torch.sum(self._neural_net.log_prob(theta, x) + torch.log(last_proposal) - self._prior.log_prob(theta))
            
            d_log_q_d_theta = grad(logq(new_theta), new_theta, create_graph=True)
            # print("d" , d_log_q_d_theta)
            
            first_deriv = torch.sum(d_log_q_d_theta[0]**2, dim=1)
            #print(first_deriv)
            second_deriv = torch.zeros(len(d_log_q_d_theta[0]))
            for i in range (theta.size(1)):
                
                second_deriv += grad(torch.sum(d_log_q_d_theta[0], dim=0)[i], new_theta, create_graph=True)[0][:,i]
            
            #print("s", second_deriv)
 
        
        return -(second_deriv+0.5*first_deriv)
    


    
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
        #if self._round > 0:
            

        return super().train(**kwargs)

