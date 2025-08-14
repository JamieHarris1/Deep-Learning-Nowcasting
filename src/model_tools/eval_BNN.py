import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import pymc as pm
from properscoring import crps_ensemble
import random
from patsy import dmatrix
import arviz as az
import matplotlib.pyplot as plt


from joblib import Parallel, delayed
from tqdm import tqdm
import time



from src.data_tools.data_utils import SeroDataset, PartialCountDataset, TrueCountDataset
from src.model_tools.train_utils import BaseTrain, SparsePropTrain, SeroTrain
from src.model_tools.models import NowcastPNN, PropPNN, SparsePropPNN, SeroPNN
from src.model_tools.evaluation import eval_pnn, eval_prop_pnn, eval_sparse_prop_pnn, eval_sero_pnn, plot_pnn_preds, plot_prop_pnn_preds, plot_sero_pnn_preds
from src.model_tools.NegativeBinomial import NegBin as NB

M = 50
D = 40
S = 1000
chains = 4

project_dir = Path.cwd()
start_year = 2022
end_year = 2022
seed = 123

# 123, 2019, 2023, 15

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Create count obj
delays_df = pd.read_csv(project_dir / "data" / "transformed" / "DENG_delays.csv")
delays_df['Collection date'] = pd.to_datetime(delays_df['Collection date'])

partial_count_dataset = PartialCountDataset(delays_df, D=D, M=M, norm=False)
true_count_dataset = TrueCountDataset(delays_df)

class PropBNNDataset(Dataset):
    def __init__(self, partial_count_obj, true_count_obj, dates):
        self.partial_count_obj = partial_count_obj
        self.true_count_obj = true_count_obj
        self.dates = dates
    
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        date = self.dates[index]
        window_dates = [date - pd.Timedelta(days=i) for i in range(M)]
        window_dates = sorted(window_dates)
        Z_obs = self.partial_count_obj.get_obs(date)
        y_true = [self.true_count_obj.get_y(day) for day in window_dates]
        z_true = [self.true_count_obj.get_z(day) for day in window_dates]
        y_true = np.array(y_true).reshape(M)
        dow = date.day_of_week
        return Z_obs, y_true, z_true, window_dates

set_seed(seed)

# End of 2023 appears to have some incomplete data
if end_year == 2023:
    dates = list(pd.date_range(f"{start_year}-01-01",f"{end_year}-12-25", freq='D'))
else:
    dates = list(pd.date_range(f"{start_year}-01-01",f"{end_year}-12-31", freq='D'))

T = len(dates)
# T = 1
    
dataset = PropBNNDataset(partial_count_dataset, true_count_dataset, dates)


def silu(x):
    return x * pm.math.sigmoid(x)


def sampler_kwargs():
    return dict(
        nuts_sampler="nutpie",
        cores=4,
        init="adapt_diag",
        chains=chains,
        draws=S,
        tune=500,
        target_accept=0.95,
        max_treedepth=10,
        nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"}
    )

def get_mask(D):
        mask_matrix = np.ones(shape=(D, D), dtype=bool)
        for i in range(D):
            for j in range(D):
                if i + j > D - 1:
                    mask_matrix[i, j] = False
        return mask_matrix

def create_fourier_features(t, n, p=10.0):
    x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)

# Constants
t = np.arange(0, M)
t_week = t % 7
t_norm = t / M

n = 14
fourier_basis_biweek = create_fourier_features(t, n=n, p=3.5)
fourier_basis_week = create_fourier_features(t, n=n, p=7)

fourier_basis_biweek = fourier_basis_biweek - fourier_basis_biweek.mean(0, keepdims=True)
fourier_basis_week = fourier_basis_week - fourier_basis_week.mean(0, keepdims=True)

spline_trend = dmatrix(
    "bs(t, df=14, degree=3, include_intercept=False)", {"t": t_norm}, return_type='dataframe'
)
X_trend = np.asarray(spline_trend)

spline_week = dmatrix(
    "cc(t_week, df=7)", {"t_week": t_week}, return_type='dataframe'
)
X_week = np.asarray(spline_week)

t_input = np.arange(M)[:, None] / M
time_input = np.concatenate([t_input, fourier_basis_biweek, fourier_basis_week], axis=1)

mask = np.ones((M,D), dtype=bool)
mask[-D:,:] = get_mask(D)


def run_PropBNN(Z_obs, Z_norm):
    with pm.Model() as PropBNN:
        log_const = pm.Normal("const", sigma=10)
    

        # Trend spline coefficients
        sigma_trend = pm.HalfNormal("sigma_trend", 10)
        beta_trend = pm.Normal("beta_trend", mu=0, sigma=sigma_trend, shape=X_trend.shape[1])
        
        # Cyclic spline coefficients (seasonality)
        sigma_week = pm.HalfNormal("sigma_week", 10)
        beta_week = pm.Normal("beta_week", mu=0, sigma=sigma_week, shape=X_week.shape[1])
        
        # GRW for prediction intervals
        init_dist = pm.Normal.dist(mu=0, sigma=1)
        sigma_rw = pm.HalfNormal("sigma_rw", 1)
        rw = pm.GaussianRandomWalk("log_lambda", sigma=sigma_rw, shape=M, init_dist=init_dist)

        # Create lam param 
        log_lam = pm.Deterministic(
                "log_lam", rw + pm.math.dot(X_trend, beta_trend) + pm.math.dot(X_week, beta_week)
            )
        lam = pm.Deterministic("lam", pm.math.exp(log_const + log_lam))

        # NN for proportions from time inputs
        n_hidden = 16
        net_sd = 0.15
        
        W1 = pm.Normal("W1", 0, net_sd, shape=(1+4*n, n_hidden))
        b1 = pm.Normal("b1", 0, net_sd, shape=(n_hidden,))
        h1 = silu(pm.math.dot(time_input, W1) + b1)

        Wz = pm.Normal("Wz", 0, net_sd, shape=(D, n_hidden))
        bz = pm.Normal("bz", 0, net_sd, shape=(n_hidden,))
        hz = silu(pm.math.dot(Z_norm, Wz) + bz)
        h1 = hz + h1
        
        # ----- Output layer: hidden → delay bins -----
        W2 = pm.Normal("W2", 0, net_sd, shape=(n_hidden, D))
        b2 = pm.Normal("b2", 0, net_sd, shape=(D,))
        p_raw = pm.math.dot(h1, W2) + b2 
        p = pm.Deterministic("p", pm.math.softmax(p_raw, axis=1))


        # Make mu param per tim/delay point
        log_mu = pm.math.log(lam[:, None]) + pm.math.log(p + 1e-6)
        mu = pm.Deterministic("mu", pm.math.exp(log_mu))

        # Overdispersion per time point
        sigma_theta = pm.HalfNormal("sigma_theta", 2)
        theta = pm.Exponential("theta", lam=1/sigma_theta, shape=(M,1))
        theta_broadcast = np.repeat(theta, D, axis=1)

        # Observed data
        z = pm.NegativeBinomial("z", mu[mask], theta_broadcast[mask], observed=Z_obs[mask])
        idata = pm.sample(progressbar=False, **sampler_kwargs())
    return idata


def run_single(idx):
    Z_obs, _, _, _ = dataset.__getitem__(idx)
    Z_norm = Z_obs / (Z_obs.max() + 1e-6)
    idata = run_PropBNN(Z_obs, Z_norm)
    mu_samples = az.extract(idata, group='posterior')['mu']
    theta_samples = az.extract(idata, group='posterior')['theta']
    return mu_samples, theta_samples
    

mu_samples = np.zeros((T, M, D, chains*S))
theta_samples = np.zeros((T, M,1,chains*S ))

start = time.time()
results = Parallel(n_jobs=1)(
    delayed(run_single)(idx) for idx in tqdm(range(T))
)
end = time.time()
print(f"Total time: {end - start:.2f} seconds")


for idx, (mu, theta) in enumerate(results):
    mu_samples[idx, :, :, :] = mu
    theta_samples[idx, :, :, :] = theta



np.savez_compressed(project_dir / "src" / "outputs" / "PropBNN" / 'mu_samples.npz',
                    mu=mu_samples.reshape(mu_samples.shape[0], -1))

np.savez_compressed(project_dir / "src" / "outputs" / "PropBNN" / 'theta_samples.npz',
                    theta=theta_samples.reshape(theta_samples.shape[0], -1))



data = np.load(project_dir / "src" / "outputs" / "PropBNN" / "mu_samples.npz")
mu_samples = data['mu'].reshape((T, M, D, chains*S))

data = np.load(project_dir / "src" / "outputs" / "PropBNN" / "theta_samples.npz")
theta_samples = data['theta'].reshape((T, M, 1, chains*S))



r = np.broadcast_to(theta_samples, mu_samples.shape)
mu = mu_samples
p = r / (r + mu)

z_samples = np.random.negative_binomial(r, p, size=mu_samples.shape)
z_samples.shape


y_pred = z_samples[:, -1, :, :].sum(1)
y_pred.shape




y_true = [dataset.__getitem__(idx)[1][-1] for idx in range(T)]
window_dates = [dataset.__getitem__(idx)[3][-1] for idx in range(T)]

y_pred_med = np.quantile(y_pred, 0.5, axis=1)
y_pred_lower = np.quantile(y_pred, 0.025, axis=1)
y_pred_upper = np.quantile(y_pred, 0.975, axis=1)

plt.plot(window_dates, y_true, label='True y', color="black")

plt.plot(window_dates, y_pred_med, label='PropBNN preds', color='green')
plt.fill_between(window_dates, y_pred_lower, y_pred_upper, color='green', alpha=0.2, label='PropBNN 95% CI')


plt.legend()
plt.tight_layout()
plt.title("PropBNN Test Performance", fontsize=16)
plt.xlabel("Date of First Symptom")
plt.ylabel("Case Count")
plt.xticks(rotation=45)
plt.show()




def compute_pica(y_true, y_pred, alpha=0.05):

    lower = np.quantile(y_pred, alpha / 2, axis=1)
    upper = np.quantile(y_pred, 1 - alpha / 2, axis=1)
    within_interval = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = np.mean(within_interval)
    expected_coverage = 1 - alpha
    return np.abs(empirical_coverage - expected_coverage)



mu = mu_samples[:, -1, :, :]
theta = theta_samples[:, -1, :, :]
theta = np.broadcast_to(theta, mu.shape)



z_true = [dataset.__getitem__(i)[2] for i in range(T)]
z_true = np.array(z_true)[:,-1, :]
z_true.shape


def log_sum_exp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


loglik = np.zeros(shape=(S*chains, T))
for s in range(chains*S):
    dist_pred = NB(torch.tensor(mu[:,:,s]), torch.tensor(theta[:,:,s]))
    log_prob = dist_pred.log_prob(torch.tensor(z_true)).sum(1)
    loglik[s, :] = log_prob


print(f"PropBNN ELPD: {log_sum_exp(loglik, axis=1).mean()}")



print(crps_ensemble(y_true, y_pred).mean())


print(compute_pica(y_true, y_pred, 0.05))