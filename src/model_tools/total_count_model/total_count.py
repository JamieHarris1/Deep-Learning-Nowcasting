from cmdstanpy import CmdStanModel


model = CmdStanModel(stan_file="latent_poisson_autoreg.stan")
fit = model.sample(data={"T": len(y), "y": y})

lambda_samples = fit.stan_variable("lambda")