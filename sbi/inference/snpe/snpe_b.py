# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Dict, Optional, Union
from copy import deepcopy

import torch
from torch import Tensor
from torch.distributions import Distribution

import sbi.utils as utils
from sbi.neural_nets.density_estimators.base import DensityEstimator
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.sbi_types import TensorboardSummaryWriter
from sbi.utils import del_entries
from torch.nn.functional import kl_div

class SNPE_B(PosteriorEstimator):
    def __init__(
        self,
        observation: Tensor, 
        bandwith: float = 0.01,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
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
        """

        self._observation = observation
        self._bandwith = bandwith
        
        
        kwargs = del_entries(locals(), entries=("self", "__class__", "observation", "bandwith"))
        super().__init__(**kwargs)

    def _log_prob_proposal_posterior(
        self, theta: Tensor, x: Tensor, masks: Tensor, proposal: Optional[Any],
    ) -> Tensor:
        """
        Return importance-weighted log probability (Lueckmann, Goncalves et al., 2017).

        Args:
            theta: Batch of parameters θ.
            x_0: Observation.
            masks: Whether to retrain with prior loss (for each prior sample).

        Returns:
            Log probability of proposal posterior.
        """
 
        batch_size = theta.shape[0]
        
        # Evaluate prior.
        #log_prob_prior = self._prior.log_prob(theta).reshape(batch_size)

        log_prob_prior = self._prior.log_prob(theta)
        utils.assert_all_finite(log_prob_prior, "prior eval.")

        # Evaluate proposal.
        log_prob_proposal = proposal.log_prob(theta)
        
        utils.assert_all_finite(log_prob_proposal, "proposal posterior eval")

        # Compute the importance weights.
        importance_weights = torch.exp(log_prob_prior-log_prob_proposal)
        # print(log_prob_prior)
        # print()
        # print(log_prob_proposal)
        return importance_weights*self._neural_net.log_prob(theta, x)


    def _calibration_kernel(self, x: Tensor):
        #print((torch.linalg.vector_norm(x-self._observation, dim=1))/self._bandwith)
        #return torch.exp(-(torch.linalg.vector_norm(x-self._observation, dim=1))/self._bandwith)
        # for x in batch_x:
        theta = self._prior.sample((100,))
        posterior_obs = deepcopy(self._posterior)
        posterior_x = deepcopy(self._posterior)
        proposal_obs = posterior_obs.set_default_x(self._observation)
        calibration_kernel = []
        for x_item in x:
            proposal_x = posterior_x.set_default_x(x_item)
            kl_divergence = kl_div(proposal_x.log_prob(theta), proposal_obs.log_prob(theta), log_target=True)
            calibration_kernel.append(torch.exp(-kl_divergence/self._bandwith))

        return torch.tensor(calibration_kernel)

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
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[Dict] = None,
    ) -> DensityEstimator:
        r"""Return density estimator that approximates the proposal posterior.

        [1] _Fast epsilon-free Inference of Simulation Models with Bayesian Conditional
            Density Estimation_, Papamakarios et al., NeurIPS 2016,
            https://arxiv.org/abs/1605.06376.

        Training is performed with maximum likelihood on samples from the latest round,
        which leads the algorithm to converge to the proposal posterior.

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
            importance_weights: The importance weights to add in the loss
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round. Not supported for
                SNPE-A.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            component_perturbation: The standard deviation applied to all weights and
                biases when, in the last round, the Mixture of Gaussians is build from
                a single Gaussian. This value can be problem-specific and also depends
                on the number of mixture components.

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
        
        if self._round > 0:
            # Compute the calibration kernel

            kwargs['calibration_kernel'] = self._calibration_kernel

            
            
        return super().train(**kwargs)
