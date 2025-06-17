data {
    int<lower=2> T;             // number of time steps
    array[T] int<lower=0> y;          // observed counts
    int<lower=0> D;             // max delay and number of days to predict over
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;
    vector<lower=0>[T] lambda;  // latent average count for observed period
}
model {
    // Priors
    alpha ~ normal(0, 5);
    beta ~ normal(0, 1);
    sigma ~ normal(0,1);
    lambda[1] ~ lognormal(0, 1);

    //Target: p(y|lambda[t])p(lambda[t]|lambda[t-1])...p(lambda[2]|lambda[1])p(lambda[1])
    
    // Latent state evolution
    for (t in 2:T) {
    target += lognormal_lpdf(lambda[t] | alpha + beta * log(lambda[t - 1]), sigma);
    }

    // Observation model
    for (t in 1:T) {
        y[t] ~ poisson(lambda[t]);
    }
}
generated quantities{
    vector<lower=0>[D] lambda_pred;  // predicted future latent average count
    array[D] int<lower=0> y_pred;  // predicted future observations

    lambda_pred[1] = lognormal_rng(alpha + beta * log(lambda[T]), sigma);
    y_pred[1] = poisson_rng(lambda_pred[1]);

    for (d in 2:D) {
        lambda_pred[d] = lognormal_rng(alpha + beta * log(lambda_pred[d - 1]), sigma);
        y_pred[d] = poisson_rng(lambda_pred[d]);
    }
}
