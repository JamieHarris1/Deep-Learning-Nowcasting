data {
    int<lower=2> T;             // number of time steps
    int<lower=0> y[T];          // observed counts
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
    vector<lower=0>[T] lambda;  // latent average count
}

model {
    // Priors
    alpha ~ normal(0, 5);
    beta ~ normal(0, 1);
    sigma ~ normal(0,1)
    lambda[1] ~ lognormal(0, 1);

    //Target: p(y|lambda[t])p(lambda[t]|lambda[t-1])...p(lambda[2]|lambda[1])p(lambda[1])
    
    // Latent state evolution
    for (t in 2:T) {
    mu = alpha + beta * log(lambda[t - 1])
    target += lognormal_lpdf(lambda[t] | mu, sigma);
    }

    // Observation model
    for (t in 1:T) {
        y[t] ~ poisson(lambda[t]);
    }
}
