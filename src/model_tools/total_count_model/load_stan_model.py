import cmdstanpy
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np


project_dir = Path.cwd()
stan_model_folder = project_dir / "src" / "outputs" / "stan_models" / "AR1_fit"

csv_files = [os.path.join(stan_model_folder, f)
             for f in os.listdir(stan_model_folder)
             if f.endswith('.csv')]

# Sort them to ensure consistent chain order (optional but recommended)
csv_files.sort()

# Load the CmdStanMCMC object from CSVs
fit = cmdstanpy.from_csv(path=csv_files)

# Extract full posterior draws
lambda_draws = fit.stan_variable('lambda')         # shape: (n_draws, T)
lambda_pred_draws = fit.stan_variable('lambda_pred')  # shape: (n_draws, D)

# Posterior means
lambda_mean = np.mean(lambda_draws, axis=0)
lambda_pred_mean = np.mean(lambda_pred_draws, axis=0)

# Credible intervals for prediction
lower_pred = np.percentile(lambda_pred_draws, 5, axis=0)
upper_pred = np.percentile(lambda_pred_draws, 95, axis=0)

# Time axes
T = len(lambda_mean[-50:])
D = len(lambda_pred_mean)
x_fitted = np.arange(1, T + 1)
x_pred = np.arange(T + 1, T + D + 1)

# Plot
plt.figure(figsize=(10, 5))

# Fitted line
plt.plot(x_fitted, lambda_mean[-50:], label='lambda[t] (fitted)', color='blue')

# Predicted line
plt.plot(x_pred, lambda_pred_mean, label='lambda_pred[d] (predicted)', color='orange')

# 90% credible interval for predictions
plt.fill_between(x_pred, lower_pred, upper_pred, color='orange', alpha=0.3, label='90% CI (pred)')

# Vertical separator
plt.axvline(x=T + 0.5, color='black', linestyle='--', label='Prediction start')

plt.xlabel('Time')
plt.ylabel('Lambda')
plt.title('Fitted and Predicted Lambda with 90% CI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()