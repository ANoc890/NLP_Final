import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, logits, tau=None, mask=None, return_soft=False, Log=False):
        if tau is None:
            tau = self.tau

        gumbel_noise = self.sample_gumbel(logits.shape).to(logits.device)
        y = logits + gumbel_noise
        y = torch.sigmoid(y / tau)

        if self.hard:
            y_hard = (y > 0.5).float()
            y = (y_hard - y).detach() + y

        if mask is not None:
            y = y * mask

        return y

"""import torch

from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret"""