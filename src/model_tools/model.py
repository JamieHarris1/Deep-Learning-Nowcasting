import torch
import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
import pandas as pd
import numpy as np
from pathlib import Path
from pyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt
from pyro.infer.mcmc.util import initialize_model

def model(x, y=None):

    lengthscale = pyro.sample("lengthscale", dist.HalfCauchy(1.0))
    variance = pyro.sample("variance", dist.InverseGamma(5.0, 1.0))
    
    kernel = gp.kernels.RBF(input_dim=1, lengthscale=lengthscale, variance=variance)

    # GPRegression with small noise for numerical stability
    gpr = gp.models.GPRegression(x, y, kernel, noise=torch.tensor(1e-3))
    gpr.model()

    # Sample latent function values f at inputs x
    f_mean, _ = gpr.forward(x)

    # Intercept term
    b0 = pyro.sample("b0", dist.Normal(5, 5))

    # Rate parameter for Poisson (squeeze needed as f is dim [N,1])
    lam = torch.exp(b0 + f_mean.squeeze(-1))
    
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Poisson(lam), obs=y)

project_dir = Path.cwd()
train_df = pd.read_csv(project_dir / "data" / "model" / "training_data.csv", index_col=0)
y = torch.tensor(train_df.sum(axis=1).values)
x = list(range(len(y)))
x = torch.tensor((x - np.min(x)) / np.max(x))


# Reshape to [N, input_dim] where input_dim=1 for GP input
x = x.unsqueeze(-1)  # shape [365, 1]

# Sample 
nuts_kernel = NUTS(model, init_strategy=pyro.infer.autoguide.initialization.init_to_median())
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=200)
mcmc.run(x, y)


# # f has shape [num_samples, N, 1]
# f = samples["f"].squeeze(-1)  # shape: [num_samples, N]
# b0 = samples["b0"].unsqueeze(-1)  # shape: [num_samples, 1]

# lam_samples = torch.exp(b0 + f)  # shape: [num_samples, N]

# lam_mean = lam_samples.mean(dim=0)
# lam_lower = lam_samples.quantile(0.025, dim=0)
# lam_upper = lam_samples.quantile(0.975, dim=0)

# t = x.squeeze().numpy()
# y_np = y.numpy()

# plt.figure(figsize=(10, 6))
# plt.plot(t, y_np, label="Actual y", color='black')
# plt.plot(t, lam_mean.numpy(), label="λ mean", color='blue')
# plt.fill_between(t, lam_lower.numpy(), lam_upper.numpy(), alpha=0.3, color='blue', label="95% CI")
# plt.legend()
# plt.xlabel("Normalized Time")
# plt.ylabel("Counts / Rate λ")
# plt.title("Posterior λ with 95% CI vs. Observed y")
# plt.tight_layout()
# plt.show()s