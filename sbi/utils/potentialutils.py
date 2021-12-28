from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch import Tensor
import torch.distributions.transforms as torch_tf
from sbi.utils.torchutils import atleast_2d, ensure_theta_batched
import numpy as np


def transformed_potential(
    theta: Union[Tensor, np.array],
    potential_fn: Callable,
    potential_tf: torch_tf.Transform,
    device: str,
    track_gradients: bool = False,
) -> Tensor:
    """Return potential after a transformation by adding the log-abs-determinant.

    In addition, this method taken care of moving the parameters to the correct device.

    Args:
        theta:  Parameters $\theta$ in transformed space.
        potential_fn: Potential function.
        potential_tf: Transformation applied before evaluating the `potential_fn`
        device: The device to which to move the parameters before evaluation.
        track_gradients: Whether or not to track the gradients of the `potential_fn`
            evaluation.
    """

    # Device is the same for net and prior.
    transformed_theta = ensure_theta_batched(
        torch.as_tensor(theta, dtype=torch.float32)
    ).to(device)
    # Transform `theta` from transformed (i.e. unconstrained) to untransformed
    # space.
    theta = potential_tf.inv(transformed_theta)
    log_abs_det = potential_tf.log_abs_det_jacobian(theta, transformed_theta)

    posterior_potential = potential_fn(theta, track_gradients=track_gradients)
    posterior_potential_transformed = posterior_potential - log_abs_det
    return posterior_potential_transformed


def pyro_potential_wrapper(theta: Dict[str, Tensor], potential: Callable) -> Callable:
    r"""Evaluate pyro-based `theta` under the negative `potential`.

        Args:
        theta: Parameters $\theta$. The tensor's shape will be
            (1, shape_of_single_theta) if running a single chain or just
            (shape_of_single_theta) for multiple chains.
        potential: Potential which to evaluate.

    Returns:
        The negative potential $-[\log r(x_o, \theta) + \log p(\theta)]$.
    """

    theta = next(iter(theta.values()))

    # Note the minus to match the pyro potential function requirements.
    return -potential(theta)