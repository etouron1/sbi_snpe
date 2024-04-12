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

ray.init


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

        # Catch invalid inputs.
        
        # if not ((density_estimator == "mdn") or callable(density_estimator)):
        #     raise TypeError(
        #         "The `density_estimator` passed to SNPE_B needs to be a "
        #         "callable or the string 'mdn'!"
        #     )
        # self.nets = []
        #self._resampling =False
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

        # log_prob_prior = torch.zeros(theta.size(0))
        # log_prob_proposal = torch.zeros(theta.size(0))

        # if torch.any(masks)!=True:
        #     # Evaluate prior
        #     log_prob_prior[torch.logical_not(masks.squeeze())] = self._prior.log_prob(theta[torch.logical_not(masks.squeeze()),:])
        #     utils.assert_all_finite(log_prob_prior, "prior eval.")

        #     # Evaluate proposal.
        #     log_prob_proposal[torch.logical_not(masks.squeeze())] = proposal.log_prob(theta[torch.logical_not(masks.squeeze()),:])
        #     utils.assert_all_finite(log_prob_proposal, "proposal posterior eval")

        # # Compute the importance weights.
        # importance_weights = torch.exp(log_prob_prior-log_prob_proposal)
        
        # if torch.any(importance_weights> 50):
        #     print()
        #     print("weight", importance_weights[importance_weights> 50])
        #     print()
        #     print("theta_0", theta[torch.logical_not(masks.squeeze()),0][importance_weights> 50])
        #     print("theta_1", theta[torch.logical_not(masks.squeeze()),1][importance_weights> 50])
        #     print()
        #     #theta_prior = self._prior.sample((1000, ))
        #     theta_proposal = proposal.sample((10000, ))
        #     #sns.kdeplot(theta_prior[:,0], label=r"prior $p$")
        #     plt.figure(figsize=(20,7))
        #     plt.subplot(121)
        #     sns.kdeplot(theta_proposal[:,0], label=r"proposal $\tilde{p}$")
        #     plt.scatter(theta[torch.logical_not(masks.squeeze()),0][importance_weights> 50], torch.zeros(len(theta[torch.logical_not(masks.squeeze()),0]))[importance_weights> 50], label=r"$\theta$ from $\tilde{p}$", color="red", marker="+")
        #     plt.title("Distribution support comparison")
        #     plt.xlabel("dim 0")
        #     plt.legend()
        #     plt.subplot(122)
        #     sns.kdeplot(theta_proposal[:,1], label=r"proposal $\tilde{p}$")
        #     plt.scatter(theta[torch.logical_not(masks.squeeze()),1][importance_weights> 50], torch.zeros(len(theta[torch.logical_not(masks.squeeze()),1]))[importance_weights> 50], label=r"$\theta$ from $\tilde{p}$", color="red", marker="+")
        #     plt.title("Distribution support comparison")
        #     plt.xlabel("dim 1")
        #     plt.legend()
        #     plt.show()
        # importance_weights = prior_theta/(self._prop_prior*prior_theta + (1-self._prop_prior)*proposal_theta)
        #if self._resampling==False:
        prop = 1.0/(self._round+1)
        prior = torch.exp(self._prior.log_prob(theta))
        
        #print(prior)
        #print("neuralnet", len(self.nets))
        #inference_method = SNPE_B(prior=self._prior, density_estimator="nsf", observation=self._observation)
        #proposal_try = prop*prior

        # for net in self.nets:
        #     proposal_try += prop*torch.exp(inference_method.build_posterior(net).set_default_x(self._observation).log_prob(theta))
        
        proposal = torch.zeros(theta.size(0))
        #mixture = torch.zeros((5000,2))
        #plot=False
        #i=1
        for density in self._proposal_roundwise:

    
            proposal += prop*torch.exp(density.log_prob(theta))
            #mixture += prop*density.sample((5000, ))
        #     if torch.max(torch.exp(density.log_prob(theta))) > 1:
        #         print("density", i)
        #         print(type(density))
        #         print(torch.exp(density.log_prob(theta))[torch.exp(density.log_prob(theta))> 1])
                
        #         plot = True
        #     if plot:
        #         sns.kdeplot(density.sample((5000, ))[:,0], label=f"proposal {i}")
        #     i+=1
        # if plot:
        #     import pandas as pd
        #     sns.kdeplot(mixture[:,0], label=f"mixture")
            
        #     plt.scatter(theta[torch.exp(density.log_prob(theta)) > 1][:,0], torch.zeros_like(theta[torch.exp(density.log_prob(theta)) > 1][:,0]), label=f'$\theta$ from proposal {i-1}$', color="red", marker="+")
        #     plt.legend()
        #     plt.show()
        #     theta_2_d = pd.DataFrame(density.sample((5000,)), columns=["0", "1"])
            
        #     sns.kdeplot(data=theta_2_d, x="0", y="1", fill=True, cbar=True)
        #     plt.show()

        
        # print(prop)
        # print("max proposal", torch.max(proposal))
        # plt.plot(theta[:,0], prior, label="prior")
        # plt.scatter(theta[:,1], proposal, label="proposal")
        # plt.scatter(theta[:,1], proposal_try, label="proposal_try", marker="+")
        # plt.legend()
        # plt.show()
        importance_weights = prior/proposal
        # ess=torch.sum(importance_weights)**2/torch.sum(importance_weights**2)
        # with open('ess.txt', 'a') as f:
        #     print('ess:', int(ess/theta.size(0)*100), file=f)
        importance_weights = importance_weights/torch.sum(importance_weights)
        # else:
        #     importance_weights = torch.ones(theta.size(0))/theta.size(0)
    
        
        #var = torch.mean((importance_weights - torch.mean(importance_weights))**2)
        #if var >10:
        #    print("variance", var)
        #ess =1/torch.sum(importance_weights**2)
        # prop_ess=int(100/(torch.sum(importance_weights**2)*theta.size(0)))
        # with open('ess.txt', 'a') as f:
        #     print('ess:', prop_ess, file=f)
        # if prop_ess < 60:
        #     if theta.size(0)>1:
        #         with open('ess.txt', 'a') as f:
        #             print("Resampling", file=f)
        #             indices = Multinomial(theta.size(0)-1, importance_weights).sample()
        #             theta = torch.index_select(input = theta,dim=0, index=indices.int())
        #             x = torch.index_select(input = x,dim=0, index=indices.int())
        #             importance_weights = torch.ones(theta.size(0))/theta.size(0)
                    
                   

                    
                    #prior_new = torch.exp(self._prior.log_prob(theta_new))
                    #proposal_new = torch.zeros(theta_new.size(0))
                    #for density in self._proposal_roundwise:
                    #    proposal_new += prop*torch.exp(density.log_prob(theta_new))
                    #importance_weights_new = prior_new/proposal_new
                    #importance_weights_new = importance_weights_new/torch.sum(importance_weights_new)
                    #ess_new=1/torch.sum(importance_weights_new**2)
                    #prop_ess_new = int(100/(torch.sum(importance_weights_new**2)*theta.size(0)))
                    #print('ess:', prop_ess_new, file=f)
                    #if  prop_ess_new > prop_ess:
                    #    print('change ess', file=f)

                    #    importance_weights = importance_weights_new 
                    #    theta = theta_new
                    #    x = torch.index_select(input = x,dim=0, index=indices.int())
        return importance_weights*self._neural_net.log_prob(theta, x)
    


    def _calibration_kernel(self, x: Tensor):
        #print((torch.linalg.vector_norm(x-self._observation, dim=1))/self._bandwith)
        #return torch.exp(-(torch.linalg.vector_norm(x-self._observation, dim=1))/self._bandwith)
        # for x in batch_x:
        
        theta = self._prior.sample((100, ))

        posterior_obs = deepcopy(self._posterior)
        posterior_x = deepcopy(self._posterior)

        proposal_obs = posterior_obs.set_default_x(self._observation)
   
        calibration_kernel = []

        for x_item in x:
            proposal_x = posterior_x.set_default_x(x_item)
            #kl_start = time.time()
            kl_divergence = kl_div(proposal_x.log_prob(theta), proposal_obs.log_prob(theta), log_target=True)
            #kl_end = time.time()
            #print("kl", kl_end-kl_start)
            calibration_kernel.append(torch.exp(-kl_divergence/self._bandwith))
        # print("sans para", calibration_kernel)
        # futures = [divergence.remote(posterior_x, proposal_obs, x_item, theta, self._bandwith) for x_item in x]
        # calibration_kernel = ray.get(futures)
        #print("para", calibration_kernel)
        return torch.tensor(calibration_kernel)

    # def kl_divergence(self, x_item, posterior_x, proposal_obs, theta):
    #     proposal_x = posterior_x.set_default_x(x_item)
    #     kl_divergence = kl_div(proposal_x.log_prob(theta), proposal_obs.log_prob(theta), log_target=True)
    #     return kl_divergence

    # def weight_for_loss(self, calibration_kernel: Callable, theta: Tensor, x: Tensor, x_0: Tensor) -> Tensor:
    #     """
    #     Return the weight for the loss composed of:
    #     - the importance-weights (ratio prior over proposal) (Lueckmann, Goncalves et al., 2017).
    #     - the calibration kernel to exclude bad simulations 
    #     (simulations x far from the observation x_0)

    #     Args:
    #         calibration_kernel: The calibraton kernel
    #         theta: Batch of parameters θ.
    #         x: Batch of corresponding data
    #         x_0: Observation.

    #     Returns:
    #         Weights for the loss.
    #     """

    #     batch_size = theta.shape[0]

    #     if calibration_kernel is None:

    #         def base_calibration_kernel(x, x_0, tau):
    #             """
    #             Define a basic calibration kernel to exclude bad simulations
    #             - if x = x_0, then the kernel is 1
    #             - else, the kernel decreases with increasing distance || x- x_0 || 
                
    #             Args:
    #                 x: Batch of simulations
    #                 x_0: The observation
    #                 tau : the bandwith of the kernel

    #             Returns:
    #                 The calibration kernel
    #             """

    #             return torch.exp(-(torch.linalg.vector_norm(x-x_0)**2)/tau**2)

    #         calibration_kernel = base_calibration_kernel

    #     x_0 = x_0.repeat(batch_size)
        
    #     # Evaluate prior.
    #     log_prob_prior = self._prior.log_prob(theta).reshape(batch_size)
    #     utils.assert_all_finite(log_prob_prior, "prior eval.")

    #     # Evaluate proposal.
    #     #log_prob_proposal = self._model_bank[-1].net.log_prob(theta, x_0)
    #     proposal = self._proposal_roundwise[-1]
    #     #utils.assert_all_finite(log_prob_proposal, "proposal posterior eval")

    #     # Compute the importance weights.
    #     importance_weights = torch.exp(log_prob_prior - log_prob_proposal)
    #     try:
    #         return importance_weights*calibration_kernel(x, x_0, 0.01)
    #     except TypeError:
    #         print("The calibration kernel must take as argument x, x_0 and a bandwith")


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
        #     batch_size = len(self._theta_roundwise[0])
        #     theta = torch.cat(self._theta_roundwise, dim=0)
            
            
        #     x = torch.cat(self._x_roundwise, dim=0)
        #     prop = 1.0/(self._round+1)
        #     prior = torch.exp(self._prior.log_prob(theta))
        #     proposal = torch.zeros(theta.size(0))
        #     for density in self._proposal_roundwise:
        #         proposal += prop*torch.exp(density.log_prob(theta))
        #     importance_weights = prior/proposal
        #     importance_weights = importance_weights/torch.sum(importance_weights)
        #     prop_ess=int(100/(torch.sum(importance_weights**2)*theta.size(0)))
        
        #     if prop_ess < 30:
        #         print("resamlping")
        #         self._resampling = True
        #         indices = Multinomial(theta.size(0)-1, importance_weights).sample()
        #         theta = torch.index_select(input = theta,dim=0, index=indices.int())
        #         eps = MultivariateNormal(torch.zeros(theta.size(1)), torch.eye(theta.size(1))*0.01).sample((theta.size(0),))
        #         theta+=eps 
        #         # theta.requires_grad=True
        #         # print("t", theta)
        #         # proposal_langevin = torch.zeros(theta.size(0))
        #         # print("init", proposal_langevin)
        #         # for density in self._proposal_roundwise:
        #         #     proposal_langevin += prop*torch.exp(density.log_prob(theta))
        #         #     print("run", density.log_prob(theta))
        #         # loss = -torch.log(proposal_langevin)
        #         # print("l", loss)
                
        #         # lmbda = 0.1
        #         # optimizer = torch.optim.SGD([theta], lr=lmbda, momentum=0.)
        #         # optimizer.zero_grad()
                
               
        #         # loss.backward()  # let autograd do its thing
        #         # optimizer.step()
        #         # eps = MultivariateNormal(torch.zeros(theta.size(1)), torch.eye(theta.size(1))*0.01).sample()
        #         # print(eps)
        #         # theta += torch.sqrt(2*lmbda)*eps
        #         x = torch.index_select(input = x,dim=0, index=indices.int())
        #         x += eps
        #         importance_weights = torch.ones(theta.size(0))/theta.size(0)
        #         self._theta_roundwise = list(torch.split(theta, batch_size))
        #         self._x_roundwise = list(torch.split(x, batch_size))
        #     self.nets.append(self._neural_net)
        #     # Compute the calibration kernel

        #    kwargs['calibration_kernel'] = self._calibration_kernel

        return super().train(**kwargs)




@ray.remote
def divergence(posterior_x, proposal_obs, x_item, theta, bandwith):
    proposal_x = posterior_x.set_default_x(x_item)
    kl_divergence = kl_div(proposal_x.log_prob(theta), proposal_obs.log_prob(theta), log_target=True)
    return torch.exp(-kl_divergence/bandwith)