from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

__all__ = ["NegBin"]


class NegBin(ExponentialFamily):
    """
    Creates a Negative Binomial distribution parameterized by mean lambda and overdispersion phi.

    Args:
        lbda [Number, Tensor]: mean of the distribution
        phi [Number, Tensor]: overdispersion parameter
    """
    arg_constraints = {"lbda": constraints.nonnegative, "phi": constraints.positive}
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return self.lbda
    
    @property
    def mode(self):
        return torch.floor((self.phi-1)*self.lbda/self.phi)
    
    def get_lbda(self):
        return self.lbda
    
    def get_phi(self):
        return self.phi

    @property
    def variance(self):
        return self.lbda + self.lbda.pow(2)/self.phi

    def __init__(self, lbda, phi, validate_args=None):
        self.lbda, self.phi = broadcast_all(lbda, phi)
        if isinstance(lbda, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.lbda.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegBin, _instance)
        batch_shape = torch.Size(batch_shape)
        new.lbda = self.lbda.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)
        super(NegBin, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    @property
    def _gamma(self):
        return torch.distributions.Gamma(
            # parameterization of alpha and beta
            concentration=self.phi,
            rate=self.phi/self.lbda#,
            #validate_args=False,
        )
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            rate = self._gamma.sample(sample_shape=sample_shape)
            return torch.poisson(rate)

    def log_prob(self, y):
        if self._validate_args:
            self._validate_sample(y)
        if len(y.size()) > 1:
            y = torch.squeeze(y)
        #lbda, phi, y = broadcast_all(self.lbda, self.phi, y)
        #lbda, phi, y = torch.broadcast_tensors(self.lbda, self.phi, y)
        lbda, phi = self.lbda, self.phi
        #print(lbda.size(), phi.size(), y.size())
        return (y + phi).lgamma() - (y+1).lgamma() - phi.lgamma() + y.xlogy(lbda) - y.xlogy((lbda + phi)) + phi.xlogy(phi) - phi.xlogy((phi + lbda))

    """@property
    def _natural_params(self):
        return (torch.log(self.rate),)

    def _log_normalizer(self, x):
        return torch.exp(x)

    @property
    def mode(self):
        return self.lbda.floor()"""


class ZINegBin(ExponentialFamily):
    """
    Zero-Inflated Negative Binomial distribution.

    Args:
        lbda [Number, Tensor]: mean of the NB component
        phi [Number, Tensor]: overdispersion parameter
        pi [Number, Tensor]: zero-inflation probability ∈ [0,1]
    """
    arg_constraints = {
        "lbda": constraints.nonnegative,
        "phi": constraints.positive,
        "pi": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    @property
    def mean(self):
        return (1 - self.pi) * self.lbda

    @property
    def variance(self):
        return (1 - self.pi) * (self.lbda + self.lbda.pow(2) / self.phi) + self.pi * (1 - self.pi) * self.lbda.pow(2)

    def __init__(self, lbda, phi, pi, validate_args=None):
        self.lbda, self.phi, self.pi = broadcast_all(lbda, phi, pi)
        if isinstance(lbda, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.lbda.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ZINegBin, _instance)
        batch_shape = torch.Size(batch_shape)
        new.lbda = self.lbda.expand(batch_shape)
        new.phi = self.phi.expand(batch_shape)
        new.pi = self.pi.expand(batch_shape)
        super(ZINegBin, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def _gamma(self):
        return torch.distributions.Gamma(
            concentration=self.phi,
            rate=self.phi / self.lbda
        )

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            shape = self.lbda.shape if not sample_shape else (*sample_shape, *self.lbda.shape)
            zero_mask = torch.bernoulli(self.pi.expand(shape))  # 1 with prob pi
            rate = self._gamma.sample(sample_shape=sample_shape)
            nb_sample = torch.poisson(rate)
            return (1 - zero_mask) * nb_sample  # zero out some samples

    def log_prob(self, y):
        if self._validate_args:
            self._validate_sample(y)
        y = torch.squeeze(y)
        lbda, phi, pi = self.lbda, self.phi, self.pi

        # NB log-prob
        nb_log_prob = (
            (y + phi).lgamma()
            - (y + 1).lgamma()
            - phi.lgamma()
            + y.xlogy(lbda)
            - y.xlogy(lbda + phi)
            + phi.xlogy(phi)
            - phi.xlogy(lbda + phi)
        )

        # Zero-inflation logic
        is_zero = (y == 0)
        log_prob_zero = torch.logsumexp(torch.stack([
            torch.log(pi + 1e-8), 
            torch.log(1 - pi + 1e-8) + nb_log_prob
        ]), dim=0)
        log_prob_nonzero = torch.log(1 - pi + 1e-8) + nb_log_prob

        return torch.where(is_zero, log_prob_zero, log_prob_nonzero)


import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import _sum_rightmost

class CustomMultinomial(Distribution):
    arg_constraints = {
        "total_count": constraints.nonnegative,
        "probs": constraints.simplex,
    }
    support = constraints.nonnegative_integer

    def __init__(self, total_count, probs, validate_args=None):
        """
        total_count: Tensor of shape (batch_size,) or scalar int
        probs: Tensor of shape (batch_size, num_categories)
        """
        self.total_count = total_count
        self.probs = probs
        batch_shape = probs.size()[:-1]

        # convert total_count to tensor if needed
        if isinstance(total_count, int):
            self.total_count = torch.tensor(total_count, device=probs.device)
        
        if self.total_count.dim() == 0:
            # scalar total_count, same for all batch elements
            batch_shape = probs.size()[:-1]
        else:
            # check batch shape compatibility
            if self.total_count.size() != batch_shape:
                raise ValueError(f"Shape mismatch: total_count {self.total_count.size()} vs probs batch {batch_shape}")

        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = sample_shape + self.batch_shape + self.probs.size()[-1:]
        samples = []

        # expand total_count and probs to sample_shape + batch_shape
        total_count_exp = self.total_count.expand(sample_shape + self.batch_shape)
        probs_exp = self.probs.expand(sample_shape + self.batch_shape + (self.probs.size(-1),))

        # flatten batch for loop efficiency
        flat_total_count = total_count_exp.reshape(-1)
        flat_probs = probs_exp.reshape(-1, self.probs.size(-1))

        for count, prob in zip(flat_total_count, flat_probs):
            dist = torch.distributions.Multinomial(total_count=count.item(), probs=prob)
            samples.append(dist.sample())
        
        samples = torch.stack(samples, dim=0)
        # reshape back to sample_shape + batch_shape + num_categories
        samples = samples.reshape(shape)
        return samples

    def log_prob(self, value):
        # value shape: sample_shape + batch_shape + num_categories
        # We do vectorized log_prob by leveraging torch.distributions.Multinomial

        # flatten batch dims to compute efficiently
        flat_value = value.reshape(-1, value.size(-1))
        flat_probs = self.probs.expand(value.shape[:-1] + (self.probs.size(-1),)).reshape(-1, self.probs.size(-1))
        flat_total_count = self.total_count.expand(value.shape[:-1]).reshape(-1)

        log_probs = []
        for val, prob, count in zip(flat_value, flat_probs, flat_total_count):
            dist = torch.distributions.Multinomial(total_count=count.item(), probs=prob)
            log_probs.append(dist.log_prob(val))
        log_probs = torch.stack(log_probs)
        return log_probs.reshape(value.shape[:-1])

    @property
    def mean(self):
        return self.total_count.unsqueeze(-1) * self.probs

    @property
    def variance(self):
        # Var[X_i] = n p_i (1-p_i), Cov[X_i, X_j] = -n p_i p_j
        n = self.total_count.unsqueeze(-1)
        p = self.probs
        var = n * p * (1 - p)
        return var
