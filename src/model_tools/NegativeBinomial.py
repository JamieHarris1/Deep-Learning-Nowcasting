import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from numbers import Number


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
        lbda, phi = self.lbda, self.phi
        return (y + phi).lgamma() - (y+1).lgamma() - phi.lgamma() + y.xlogy(lbda) - y.xlogy((lbda + phi)) + phi.xlogy(phi) - phi.xlogy((phi + lbda))



