seed = 43
obs_N = 50000
beta_N = 101
sample_N = 1000

beta_actual = round(rnorm(beta_N, 0, 1), 2)
beta_actual[sample(1:beta_N, 20)] = 0

datX = NULL
for (i in 1:beta_N) {
  x = round(rnorm(obs_N, 0, 1), 2)
  datX = cbind(datX, x)
}
datX[,1] = 1
colnames(datX) = paste("x", 0:(beta_N-1), sep="")

###############################################
# No Regularization: Using regression formula #
###############################################
beta_estimate = NULL
for (sample_cnt in 1:sample_N) {
  eps = rnorm(obs_N, 0, 0.25)
  y = datX %*% beta_actual + eps

  lmfit = lm(y ~ datX[,-1])
  beta_estimate = rbind(beta_estimate, lmfit$coeff)
  
  print(sample_cnt)
}
colnames(beta_estimate) = paste("beta", 0:(beta_N-1), sep="")

bias = colMeans(beta_estimate) - beta_actual
mean(bias) # 3.31994e-06

#############################################
# No Regularization: Using optim() function #
#############################################
seed = 43
lossFunc = function(beta) {
  loss_total = sum((y - datX %*% beta)^2)
  return(loss_total)
}

beta_estimate = NULL
for (sample_cnt in 1:sample_N) {
  eps = rnorm(obs_N, 0, 0.25)
  y = datX %*% beta_actual + eps

  optim_out = optim(par=rnorm(beta_N, 0, 1), fn=lossFunc)
  beta_estimate = rbind(beta_estimate, optim_out$par)
  
  print(sample_cnt)
}
colnames(beta_estimate) = paste("beta", 0:(beta_N-1), sep="")

bias = colMeans(beta_estimate) - beta_actual
mean(bias) # 0.00361135

#########
# Lasso #
#########
seed = 43
lossFunc_Lasso = function(allParams) {
  lambda = allParams[1]
  beta = allParams[2:length(allParams)]
  loss_total = sum((y - datX %*% beta)^2) + lambda*sum(abs(beta))
  return(loss_total)
}

beta_estimate = NULL
for (sample_cnt in 1:sample_N) {
  eps = rnorm(obs_N, 0, 0.1)
  y = datX %*% beta_actual + eps

  optim_out = optim(par=rnorm(beta_N + 1, 0, 1), fn=lossFunc_Lasso)
  beta_estimate = rbind(beta_estimate, optim_out$par[2:length(optim_out$par)])
  
  print(sample_cnt)
}
colnames(beta_estimate) = paste("beta", 0:(beta_N-1), sep="")

bias = colMeans(beta_estimate) - beta_actual
mean(bias) # 0.00234412
