library(dplyr)
library(nimble)
library(coda)
library(mgcv)
library(viridis)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(abind)
library(doParallel)
library(tidyverse)
library(data.table)

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

glm_inits_func=function(seed,z,D,N,C,K_t,K_w){
  
  set.seed(seed)
  
  inits=list(y=rep(NA,C),z=matrix(NA,nrow=C,ncol=D),theta=rexp(1,0.01),
             iota=rnorm(1,mean(log(z[z>0]),na.rm=T),0.1),psi=rnorm(D,apply(log(z),2,function(x)mean(x[x>0],na.rm=T)),1),
             tau_alpha=rinvgamma(1,0.5,0.5),tau_eta=rinvgamma(1,0.5,0.5),
             tau_beta=rinvgamma(D,0.5,0.5),kappa_alpha=rnorm(K_t,0,0.1),
             kappa_eta=rnorm(K_w,0,0.1),kappa_beta=matrix(rnorm(K_t*D,0,0.1),nrow=K_t,ncol=D))
  
  for(t in 1:C){
    for(d in 1:(D)){
      if(is.na(z[t,d])) inits$z[t,d]=rpois(1,median(z[,d],na.rm=T))
    }
  }
  return(inits)
}


run_glm=function(D,N,C,n_knots=c(floor(N/10),13),censored_dengue,niter,nburnin,thin,nchains,seed){
  
  
  # Set data and inits on the model
  glm_data=list(z=cbind(censored_dengue[1:C,1:D],
                        rowSums(censored_dengue[1:C,])-rowSums(censored_dengue[1:C,1:D])),
                X_t=blank_jagam$jags.data$X[,2:(n_knots[1])],S_t=blank_jagam$jags.data$S1,
                X_w=blank_jagam$jags.data$X[,(n_knots[1]+1):(n_knots[1]+n_knots[2]-2)],S_w=blank_jagam$jags.data$S2,
                zeros=rep(0,max(glm_constants$K_t,glm_constants$K_w)),w=week_index[1:N])
  
  # Generate initial values.
  glm_inits=list(c1=glm_inits_func(seed[1],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
                 c2=glm_inits_func(seed[2],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
                 c3=glm_inits_func(seed[3],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
                 c4=glm_inits_func(seed[4],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2))
  glm_compiled_model$setData(glm_data)
  output=list()
  
  # Run the MCMC.
  output$mcmc=runMCMC(glm_compiled_mcmc,inits=glm_inits[1:nchains],nchains=nchains,
                      niter=niter,nburnin=nburnin,thin=thin,samplesAsCodaMCMC = TRUE,
                      setSeed = seed[1:nchains])
  
  if (!is.list(output$mcmc)) {
    output$mcmc <- list(output$mcmc)
  }
  
  combined_samples=do.call('rbind',output$mcmc)
  output$samples=list()
  output$index=list()
  n_sim=dim(combined_samples)[1]
  
  output$index$alpha=which(dimnames(combined_samples)[[2]]=='alpha[1]'):which(dimnames(combined_samples)[[2]]==paste('alpha[',N,']',sep=''))
  output$samples$alpha <- array(combined_samples[,output$index$alpha],dim=c(n_sim,N))
  
  output$index$eta=which(dimnames(combined_samples)[[2]]=='eta[1]'):which(dimnames(combined_samples)[[2]]==paste('eta[',52,']',sep=''))
  output$samples$eta=array(combined_samples[,output$index$eta],dim=c(n_sim,52))
  
  output$index$beta=which(dimnames(combined_samples)[[2]]=='beta[1, 1]'):which(dimnames(combined_samples)[[2]]==paste('beta[',N,', ',D,']',sep=''))
  output$samples$beta=array(combined_samples[,output$index$beta],dim=c(n_sim,N,D))
  
  output$index$mu=which(dimnames(combined_samples)[[2]]=='mu[1, 1]'):which(dimnames(combined_samples)[[2]]==paste('mu[',N,', ',D+1,']',sep=''))
  output$samples$mu=array(combined_samples[,output$index$mu],dim=c(n_sim,N,D+1))
  
  output$index$lambda=which(dimnames(combined_samples)[[2]]=='lambda[1]'):which(dimnames(combined_samples)[[2]]==paste('lambda[',N,']',sep=''))
  output$samples$lambda <- array(combined_samples[,output$index$lambda],dim=c(n_sim,N))
  
  output$index$theta=which(dimnames(combined_samples)[[2]]=='theta')
  output$samples$theta=combined_samples[,output$index$theta]
  
  output$index$z=which(dimnames(combined_samples)[[2]]=='z[1, 1]'):which(dimnames(combined_samples)[[2]]==paste('z[',C,', ',D+1,']',sep=''))
  output$samples$z=array(NA,dim=c(n_sim,N,D+1))
  output$samples$z[,1:C,]=array(combined_samples[,output$index$z],dim=c(n_sim,C,D+1))
  output$samples$z[,(C+1):N,]=array(rnbinom(n_sim*(N-C)*(D+1),size=rep(output$samples$theta,(N-C)*(D+1)),
                                            mu=output$samples$mu[,(C+1):N,]),dim=c(n_sim,N-C,D+1))
  
  output$samples$y=rowSums(output$samples$z,dims=2)
  
  # Simulate posterior replicates.
  
  output$replicates=list()
  
  output$replicates$z=array(rnbinom(n_sim*(D+1)*C,mu=output$samples$mu[,1:C,],size=rep(output$samples$theta,C*(D+1))),dim = c(n_sim,C,D+1))
  
  output$replicates$y=rowSums(output$replicates$z,dims=2)
  return(output)
}


n_knots <- c(18,9)
D = 39
N <- 115
C <- 114
# Random number generator seed.
seed=c(17274757,52365823,34576435,16797945)
Time = 365
base_path <- "/Users/jamieharris/Documents/GitHub/Imperial/Dengue-Nowcasting-Thesis/data/model/GLM_data"
save_base_path <- "/Volumes/GLENN_SSD/GLM_outputs"

input_file <- paste0(base_path, "/Z_obs_", 0, ".csv")
data <- read.csv(input_file, header = FALSE, na.strings = c("NA", "<NA>"))
dengue<- as.matrix(data)
dengue <- rbind(dengue, rep(NA, ncol(dengue)))
reduced_dengue <- dengue[1:N,]
full_y <- apply(dengue,1,sum)
dengue_y <- apply(reduced_dengue,1,sum)
censored_dengue <- reduced_dengue
censored_dengue[outer(1:dim(reduced_dengue)[1], 0:(dim(reduced_dengue)[2]-1), FUN = "+") > C] <- NA


set.seed(seed[1])
week_index=rep(1:52,10)

# Set up splines.
blank_data=data_frame(y=rnorm(N,0,1),t=1:N,w=week_index[1:N])
blank_jagam=jagam(y~s(t,bs='cs',k=n_knots[1])+s(w,bs='cc',k=n_knots[2]),data=blank_data,file='blank.jags',
                  knots=list(t=seq(1,C,length=n_knots[1]),w=seq(0,52,length=n_knots[2])))

# Model code (quasi-BUGS language).
glm_code=nimbleCode({
  for(t in 1:N){
    for(d in 1:D){
      log(mu[t,d]) <- psi[d] + alpha[t] + beta[t,d] + eta[w[t]] 
    }
    lambda[t] <- sum(mu[t,1:D])
  }
  for(t in 1:C){
    for(d in 1:D){
      # Model for partials
      z[t,d] ~ dnegbin(p[t,d],theta)
      p[t,d] <- theta/(theta+mu[t,d])
    }
    y[t] <- sum(z[t,1:D])
  }
  # Temporal spline
  Omega_alpha[1:K_t,1:K_t] <- S_t[1:K_t,1:K_t]*tau_alpha
  kappa_alpha[1:K_t] ~ dmnorm(zeros[1:K_t],Omega_alpha[1:K_t,1:K_t])
  alpha[1:N] <- X_t[1:N,1:K_t]%*%kappa_alpha[1:K_t]
  # Seasonal spline
  Omega_eta[1:K_w,1:K_w] <- S_w[1:K_w,1:K_w]*tau_eta
  kappa_eta[1:K_w] ~ dmnorm(zeros[1:K_w],Omega_eta[1:K_w,1:K_w])
  eta[1:52] <- X_w[1:52,1:K_w]%*%kappa_eta[1:K_w]
  # Delay splines
  for(d in 1:D){
    Omega_beta[1:K_t,1:K_t,d] <- S_t[1:K_t,1:K_t]*tau_beta[d]
    kappa_beta[1:K_t,d] ~ dmnorm(zeros[1:K_t],Omega_beta[1:K_t,1:K_t,d])
    beta[1:N,d] <- X_t[1:N,1:K_t]%*%kappa_beta[1:K_t,d]
  }
  # Prior distributions
  iota ~ dnorm(0,sd=10)
  tau_alpha ~ dinvgamma(0.5,0.5)
  tau_eta ~ dinvgamma(0.5,0.5)
  theta ~ dexp(0.01) # Negative Binomial dispersion
  for(d in 1:D){
    psi[d] ~ dnorm(iota,sd=10)
    tau_beta[d] ~ dinvgamma(0.5,0.25)
  }
})

# Data for the model.
glm_constants=list(N=N,C=C,D=D+1,K_t=n_knots[1]-1,K_w=n_knots[2]-2)
glm_data=list(z=cbind(censored_dengue[1:C,1:D],
                      rowSums(censored_dengue[1:C,])-rowSums(censored_dengue[1:C,1:D])),
              X_t=blank_jagam$jags.data$X[,2:(n_knots[1])],S_t=blank_jagam$jags.data$S1,
              X_w=blank_jagam$jags.data$X[,(n_knots[1]+1):(n_knots[1]+n_knots[2]-2)],S_w=blank_jagam$jags.data$S2,
              zeros=rep(0,max(glm_constants$K_t,glm_constants$K_w)),w=week_index[1:N])

# Generate initial values.
glm_inits=list(c1=glm_inits_func(seed[1],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
               c2=glm_inits_func(seed[2],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
               c3=glm_inits_func(seed[3],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2),
               c4=glm_inits_func(seed[4],glm_data$z,D+1,N,C,n_knots[1]-1,n_knots[2]-2))

# Build the model.
glm_model=nimbleModel(glm_code,glm_constants,glm_data,glm_inits)

glm_compiled_model=compileNimble(glm_model,resetFunctions = TRUE)

glm_mcmc_config=configureMCMC(glm_model,monitors=c('iota','psi','alpha','beta','eta','theta','lambda','mu',
                                                   'kappa_alpha','tau_alpha','kappa_eta','tau_eta','kappa_beta','tau_beta',
                                                   'y','z'),useConjugacy = FALSE)

glm_mcmc=buildMCMC(glm_mcmc_config)
glm_compiled_mcmc=compileNimble(glm_mcmc,resetFunctions=TRUE)


gc()
chain <- 1
start_time <- Sys.time()
print(start_time)
for(idx in 0:(Time-1)){
  input_file <- paste0(base_path, "/Z_obs_", idx, ".csv")
  data <- read.csv(input_file, header = FALSE, na.strings = c("NA", "<NA>"))
  dengue<- as.matrix(data)
  dengue <- rbind(dengue, rep(NA, ncol(dengue)))
  reduced_dengue <- dengue[1:N,]
  full_y <- apply(dengue,1,sum)
  dengue_y <- apply(reduced_dengue,1,sum)
  censored_dengue <- reduced_dengue
  censored_dengue[outer(1:dim(reduced_dengue)[1], 0:(dim(reduced_dengue)[2]-1), FUN = "+") > C] <- NA
  
  # output <- glm_model=run_glm(D,N,C,n_knots=c(18,9),censored_dengue,niter=60000,nburnin=1000,thin=5,
  #                             nchains=1,seed=seed)
  output <- run_glm(D,N,C,n_knots=c(18,9),censored_dengue,niter=51000,nburnin=1000,thin=5,
                              nchains=1,seed=seed)
  y_samples <- output$replicates$y[,114]
  percentile_99 <- quantile(y_samples, 0.95)
  
  # Remove the values above the 99th percentile
  y_samples <- y_samples[y_samples < percentile_99]
  save_path <- paste0(save_base_path, "/chain_", chain)
  fwrite(as.data.frame(y_samples), file = file.path(save_path, paste0("/y_pred_", idx, ".csv")), na = "NA")
  gc()
}
end_time <- Sys.time()
print(paste0("Run for ", Time, " time points"))
end_time - start_time

