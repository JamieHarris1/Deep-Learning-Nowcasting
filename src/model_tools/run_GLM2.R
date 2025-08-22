suppressPackageStartupMessages({
  
  
  library(dplyr)
  library(nimble)
  library(coda)
  library(mgcv)
  library(viridis)
  library(ggplot2)
  library(gridExtra)
  library(reshape2)
  # library(ggfan)
  library(abind)
  library(doParallel)
  library(tidyverse)
  library(data.table)
  library(doParallel)
  library(foreach)
  
})


D=40
N <- 114
C <- 114
n_knots=c(18,9)
nchains <- 1
seed <-  c(123,124,125,126)


base_path <- "/Users/jamieharris/Documents/GitHub/Imperial/Dengue-Nowcasting-Thesis/data/model/GLM_data"
save_base_path <- "/Volumes/GLENN_SSD/GLM_outputs"

input_file <- paste0(base_path, "/Z_obs_", 0, ".csv")
data <- read.csv(input_file, header = FALSE, na.strings = c("NA", "<NA>"))
data <- as.matrix(data)

gdm_2_inits_func=function(seed,z,y,D,N,C,K_t,K_w){
  
  set.seed(seed)
  
  dispersion <- var(z, na.rm = TRUE) / mean(z, na.rm = TRUE) - 1
  inits=list(y=rep(NA,C),z=matrix(NA,nrow=C,ncol=D),theta=rexp(1,dispersion),
             iota=rnorm(1,mean(log(y),na.rm=T),0.1),beta=sort(rnorm(D,0,1)),phi=rexp(D,0.01),
             tau_alpha=rinvgamma(1,0.5,0.5),tau_eta=rinvgamma(1,0.5,0.5),
             tau_psi=rinvgamma(1,0.5,0.5),kappa_alpha=rnorm(K_t,0,0.1),
             kappa_eta=rnorm(K_w,0,0.1),kappa_psi=rnorm(K_t,0,0.1))
  
  for(t in 1:C){
    for(d in 1:D){
      if(is.na(z[t,d])) inits$z[t,d]=rpois(1,median(z[,d],na.rm=TRUE))
    }
    
    if(is.na(y[t])) inits$y[t]=sum(c(inits$z[t,],z[t,],rpois(1,median(y-rowSums(z),na.rm=T))),na.rm=TRUE)
  }
  return(inits)
}

dbetabin=nimbleFunction(run=function(x=double(0),mu=double(0),phi=double(0),size=double(0),log=integer(0)){
  returnType(double(0))
  if(x>=0&x<=size){
    return(lgamma(size+1)+lgamma(x+mu*phi)+lgamma(size-x+(1-mu)*phi)+lgamma(phi)-
             lgamma(size+phi)-lgamma(mu*phi)-lgamma((1-mu)*phi)-lgamma(size-x+1)-lgamma(x+1))
  }else{
    return(-Inf)
  }
})

rbetabin=nimbleFunction(run=function(n=integer(0),mu=double(0),phi=double(0),size=double(0)){
  pi=rbeta(1,mu*phi,(1-mu)*phi)
  returnType(double(0))
  return(rbinom(1,size,pi))
})

registerDistributions(list(dbetabin=list(
  BUGSdist='dbetabin(mu,phi,size)',discrete=TRUE)))

week_index=rep(1:52,40)


# Set up splines.
blank_data=data_frame(y=rnorm(N,0,1),t=1:N,w=week_index[1:N])
blank_jagam=jagam(y~s(t,bs='cs',k=n_knots[1])+s(w,bs='cc',k=n_knots[2]),data=blank_data,file='blank.jags',
                  knots=list(t=seq(1,C,length=n_knots[1]),w=seq(0,52,length=n_knots[2])))


# Model code (quasi-BUGS language).
gdm_2_code=nimbleCode({
  for(t in 1:N){
    log(lambda[t]) <- alpha[t] + eta[w[t]] 
    for(d in 1:D){
      probit(p[t,d]) <- beta[d]+psi[t]
    }
    nu[t,1] <- p[t,1]
    for(d in 2:D){
      nu[t,d] <- (p[t,d]-p[t,d-1])/(1-p[t,d-1])
    }
  }
  for(t in 1:C){
    # Model for total counts
    y[t] ~ dnegbin(theta/(theta+lambda[t]),theta)
    # Model for delayed counts
    z[t,1] ~ dbetabin(nu[t,1],phi[1],y[t])
    for(d in 2:D){
      z[t,d] ~ dbetabin(nu[t,d],phi[d],y[t]-sum(z[t,1:(d-1)]))
    }
  }
  # Temporal spline
  Omega_alpha[1:K_t,1:K_t] <- S_t[1:K_t,1:K_t]*tau_alpha
  kappa_alpha[1:K_t] ~ dmnorm(zeros[1:K_t],Omega_alpha[1:K_t,1:K_t])
  alpha[1:N] <- rep(iota,N) + X_t[1:N,1:K_t]%*%kappa_alpha[1:K_t]
  # Seasonal spline
  Omega_eta[1:K_w,1:K_w] <- S_w[1:K_w,1:K_w]*tau_eta
  kappa_eta[1:K_w] ~ dmnorm(zeros[1:K_w],Omega_eta[1:K_w,1:K_w])
  eta[1:52] <- X_w[1:52,1:K_w]%*%kappa_eta[1:K_w]
  # Delay spline
  Omega_psi[1:K_t,1:K_t] <- S_t[1:K_t,1:K_t]*tau_psi
  kappa_psi[1:K_t] ~ dmnorm(zeros[1:K_t],Omega_psi[1:K_t,1:K_t])
  psi[1:N] <- X_t[1:N,1:K_t]%*%kappa_psi[1:K_t]
  # Prior distributions
  iota ~ dnorm(0,sd=10)
  tau_alpha ~ dinvgamma(0.5,0.5)
  tau_eta ~ dinvgamma(0.5,0.5)
  tau_psi ~ dinvgamma(0.5,0.25)
  for(d in 1:D){
    phi[d] ~ dexp(0.01)
  }
  beta[1] ~ dnorm(0,sd=10)
  for(d in 2:D){
    beta[d] ~ T(dnorm(beta[d-1],sd=10),beta[d-1],)
  }
  theta ~ dexp(0.01) # Negative Binomial dispersion
})

# Data for the model.
gdm_2_constants=list(N=N,C=C,D=D,K_t=n_knots[1]-1,K_w=n_knots[2]-2)
gdm_2_data=list(z=data[1:C,1:D],y=rowSums(data)[1:C],
                X_t=blank_jagam$jags.data$X[,2:(n_knots[1])],S_t=blank_jagam$jags.data$S1,
                X_w=blank_jagam$jags.data$X[,(n_knots[1]+1):(n_knots[1]+n_knots[2]-2)],S_w=blank_jagam$jags.data$S2,
                zeros=rep(0,max(gdm_2_constants$K_t,gdm_2_constants$K_w)),w=week_index[1:N])


# Generate initial values.
gdm_2_inits=list(c1=gdm_2_inits_func(seed[1],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                 c2=gdm_2_inits_func(seed[2],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                 c3=gdm_2_inits_func(seed[3],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                 c4=gdm_2_inits_func(seed[4],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2))

# Build the model.
gdm_2_model=nimbleModel(gdm_2_code,gdm_2_constants,gdm_2_data,gdm_2_inits)

gdm_2_compiled_model=compileNimble(gdm_2_model)

gdm_2_mcmc_config=configureMCMC(gdm_2_model,monitors=c('theta','nu', 'lambda',
                                                       'phi','y','z'),useConjugacy = FALSE)
gdm_2_mcmc=buildMCMC(gdm_2_mcmc_config)
gdm_2_compiled_mcmc=compileNimble(gdm_2_mcmc,project=gdm_2_model)




run_glm_iter <- function(idx, chain, seed){
    set.seed(seed[1])
    input_file <- paste0(base_path, "/Z_obs_", idx, ".csv")
    data <- read.csv(input_file, header = FALSE, na.strings = c("NA", "<NA>"))
    data <- as.matrix(data)
    
    # Construct gdm_data
    gdm_2_data=list(z=data[1:C,1:D],y=rowSums(data)[1:C],
                    X_t=blank_jagam$jags.data$X[,2:(n_knots[1])],S_t=blank_jagam$jags.data$S1,
                    X_w=blank_jagam$jags.data$X[,(n_knots[1]+1):(n_knots[1]+n_knots[2]-2)],S_w=blank_jagam$jags.data$S2,
                    zeros=rep(0,max(gdm_2_constants$K_t,gdm_2_constants$K_w)),w=week_index[1:N])
    
    
    # Generate initial values.
    gdm_2_inits=list(c1=gdm_2_inits_func(seed[1],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                     c2=gdm_2_inits_func(seed[2],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                     c3=gdm_2_inits_func(seed[3],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2),
                     c4=gdm_2_inits_func(seed[4],data[1:C,1:D],rowSums(data)[1:C],D,N,C,n_knots[1]-1,n_knots[2]-2))
    
    # Set data and inits on the model
    gdm_2_compiled_model$setData(gdm_2_data)
    
    # Run MCMC
    output <- list()
    output$mcmc=runMCMC(gdm_2_compiled_mcmc,inits=gdm_2_inits[1:nchains],nchains=nchains,
                        niter=post_samples+burnin,nburnin=burnin,thin=thin_samples,samplesAsCodaMCMC = TRUE,
                        setSeed = seed[1:nchains])
    
    # For running a single chain
    if (!is.list(output$mcmc)) {
      output$mcmc <- list(output$mcmc)
    }
    
    combined_samples=do.call('rbind',output$mcmc)
    output$samples=list()
    output$index=list()
    n_sim=dim(combined_samples)[1]
    
    
    output$index$theta=which(dimnames(combined_samples)[[2]]=='theta')
    output$samples$theta=combined_samples[,output$index$theta]
    
    output$index$lambda=which(dimnames(combined_samples)[[2]]=='lambda[1]'):which(dimnames(combined_samples)[[2]]==paste('lambda[',N,']',sep=''))
    output$samples$lambda <- array(combined_samples[,output$index$lambda],dim=c(n_sim,N))
    
    output$index$nu=which(dimnames(combined_samples)[[2]]=='nu[1, 1]'):which(dimnames(combined_samples)[[2]]==paste('nu[',N,', ',D,']',sep=''))
    output$samples$nu=array(combined_samples[,output$index$nu],dim=c(n_sim,N,D))
    
    output$index$phi=which(dimnames(combined_samples)[[2]]=='phi[1]'):which(dimnames(combined_samples)[[2]]==paste('phi[',D,']',sep=''))
    output$samples$phi=array(combined_samples[,output$index$phi],dim=c(n_sim,D))
    
    output$index$z=which(dimnames(combined_samples)[[2]]=='z[1, 1]'):which(dimnames(combined_samples)[[2]]==paste('z[',C,', ',D,']',sep=''))
    output$samples$z=array(combined_samples[,output$index$z],dim=c(n_sim,C,D))
    
    output$samples$xi=output$samples$nu*array(output$samples$phi[,sort(rep(1:D,N))],dim=c(n_sim,N,D))
    output$samples$omega=(1-output$samples$nu)*array(output$samples$phi[,sort(rep(1:D,N))],dim=c(n_sim,N,D))

    # Simulate posterior replicates.
    
    output$replicates=list()
    
    output$replicates$y=matrix(rnbinom(n_sim*C,mu=output$samples$lambda[,1:C],size=output$samples$theta),nrow=n_sim)
    
    output$replicates$p=array(NA,dim=c(n_sim,C,D))
    for(t in 1:C){
      for(d in 1:D){
        output$replicates$p[,t,d]=rbeta(n_sim,output$samples$xi[,t,d],output$samples$omega[,t,d])
      }
    }
    
    output$replicates$z=array(NA,dim = c(n_sim,C,D))
    output$replicates$z[,,1]=matrix(rbinom(n_sim*C,size=output$replicates$y,prob=output$replicates$p[,,1]),nrow=n_sim)
    output$replicates$z[,,2]=matrix(rbinom(n_sim*C,size=output$replicates$y-output$replicates$z[,,1],prob=output$replicates$p[,,2]),nrow=n_sim)
    for(d in 3:D){
      output$replicates$z[,,d]=matrix(rbinom(n_sim*C,size=output$replicates$y-rowSums(output$replicates$z[,,1:(d-1)],dims=2),
                                             prob=output$replicates$p[,,d]),nrow=n_sim)
    }
    z_pred <- output$samples$z
    return(output)
    y_pred <- output$replicates$y[, 114]
    save_path <- paste0(save_base_path, "/chain_", chain)
    fwrite(as.data.frame(y_pred), file = file.path(save_path, paste0("/y_pred_", idx, ".csv")), na = "NA")
}

Time = 1
burnin <- 1000
post_samples <- 1000
thin_samples <- 1
start_time <- Sys.time()
print(start_time)
# Run 1 chain
for(idx in 0:(Time-1)){
  output <- run_glm_iter(idx, chain=1, seed=seed)
}

# results <- mclapply(0:(Time-1), run_glm_iter, chain=1, seed=seed, mc.cores = 4)
# results <- mclapply(0:(Time-1), run_glm_iter, chain=2, seed=seed, mc.cores = 4)
# results <- mclapply(0:(Time-1), run_glm_iter, chain=3, seed=seed, mc.cores = 4)
# results <- mclapply(0:(Time-1), run_glm_iter, chain=4, seed=seed, mc.cores = 4)
end_time <- Sys.time()
print(paste0("Run for ", Time, " time points"))
end_time - start_time



