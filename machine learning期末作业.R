
# Bayesian Logistic Regression (ML-style)
# MCMC + Train/Test Split + Performance Metrics

rm(list = ls())
set.seed(123)


# 1. 数据准备

library(MASS)

data(Pima.tr)
data(Pima.te)

data_all <- rbind(Pima.tr, Pima.te)


y <- as.numeric(data_all$type == "Yes")

X <- scale(as.matrix(data_all[,1:7]))

n <- nrow(X)
p <- ncol(X)

# 2. 机器学习范式：Train / Test 划分（70/30）
set.seed(2025)
idx <- sample(1:n, size = floor(0.7 * n))

X_train <- X[idx, ]
y_train <- y[idx]

X_test  <- X[-idx, ]
y_test  <- y[-idx]


# 3. Log-posterior（基于训练集）

log_posterior <- function(beta, X, y, sigma2) {
  eta <- X %*% beta
  loglik <- sum(y * eta - log(1 + exp(eta)))
  logprior <- sum(dnorm(beta, 0, sqrt(sigma2), log = TRUE))
  loglik + logprior
}


# 4. MCMC 参数设置


n_iter <- 12000
burnin <- 2000

a <- 2
b <- 2

beta <- rep(0, p)
sigma2 <- 1

beta_chain <- matrix(NA, n_iter, p)
sigma2_chain <- numeric(n_iter)

accept <- 0
proposal_sd <- 0.15


# 5. MCMC 主循环
for (t in 1:n_iter) {
  
  beta_prop <- beta + rnorm(p, 0, proposal_sd)
  
  log_ratio <- log_posterior(beta_prop, X_train, y_train, sigma2) -
    log_posterior(beta, X_train, y_train, sigma2)
  
  if (log(runif(1)) < log_ratio) {
    beta <- beta_prop
    accept <- accept + 1
  }
  
  shape <- a + p / 2
  rate  <- b + sum(beta^2) / 2
  sigma2 <- 1 / rgamma(1, shape = shape, rate = rate)
  
  beta_chain[t, ] <- beta
  sigma2_chain[t] <- sigma2
}

cat("MH acceptance rate:", accept / n_iter, "\n")


# 6. 后验样本


beta_post  <- beta_chain[(burnin+1):n_iter, ]
sigma2_post <- sigma2_chain[(burnin+1):n_iter]

beta_mean <- colMeans(beta_post)
beta_ci <- apply(beta_post, 2, quantile, c(0.025, 0.975))

beta_table <- data.frame(
  feature = colnames(X),
  mean = beta_mean,
  low  = beta_ci[1,],
  high = beta_ci[2,]
)

print(beta_table)


# 7. 机器学习评估：测试集预测与性能指标


# 后验预测概率（测试集）
p_test <- apply(beta_post, 1, function(b) {
  1 / (1 + exp(-X_test %*% b))
})

# 后验均值预测
p_hat <- rowMeans(p_test)

# Accuracy
pred <- ifelse(p_hat > 0.5, 1, 0)
acc <- mean(pred == y_test)

# LogLoss
logloss <- -mean(y_test * log(p_hat) + (1 - y_test) * log(1 - p_hat))

cat("\n=== ML Evaluation on Test Set ===\n")
cat("Test Accuracy:", round(acc,4), "\n")
cat("Test LogLoss:", round(logloss,4), "\n")


# 8. 对比：频率学派 Logistic（ML baseline）
df_train <- data.frame(y = y_train, X_train)
df_test  <- data.frame(y = y_test,  X_test)

fit_mle <- glm(y ~ ., data = df_train, family = binomial())

p_mle <- predict(fit_mle, newdata = df_test, type = "response")

pred_mle <- ifelse(p_mle > 0.5, 1, 0)
acc_mle <- mean(pred_mle == y_test)
logloss_mle <- -mean(y_test * log(p_mle) + (1 - y_test) * log(1 - p_mle))

cat("\n=== MLE Logistic Baseline ===\n")
cat("Test Accuracy:", round(acc_mle,4), "\n")
cat("Test LogLoss:", round(logloss_mle,4), "\n")

# Repeated random split evaluation 


set.seed(2026)

n_repeat <- 10

acc_bayes <- numeric(n_repeat)
ll_bayes  <- numeric(n_repeat)

acc_mle <- numeric(n_repeat)
ll_mle  <- numeric(n_repeat)

for (r in 1:n_repeat) {
  
  # ---- 随机划分 ----
  idx <- sample(1:n, size = floor(0.7 * n))
  X_train <- X[idx, ]
  y_train <- y[idx]
  X_test  <- X[-idx, ]
  y_test  <- y[-idx]
  
  # ---- MCMC 初始化 ----
  beta <- rep(0, p)
  sigma2 <- 1
  
  beta_chain <- matrix(NA, n_iter, p)
  accept <- 0
  
  # ---- MCMC（训练集） ----
  for (t in 1:n_iter) {
    
    beta_prop <- beta + rnorm(p, 0, proposal_sd)
    
    log_ratio <- log_posterior(beta_prop, X_train, y_train, sigma2) -
      log_posterior(beta, X_train, y_train, sigma2)
    
    if (log(runif(1)) < log_ratio) {
      beta <- beta_prop
      accept <- accept + 1
    }
    
    shape <- a + p / 2
    rate  <- b + sum(beta^2) / 2
    sigma2 <- 1 / rgamma(1, shape = shape, rate = rate)
    
    beta_chain[t, ] <- beta
  }
  
  beta_post <- beta_chain[(burnin+1):n_iter, ]
  
  # ---- 贝叶斯测试集预测 ----
  p_test <- apply(beta_post, 1, function(b)
    1 / (1 + exp(-X_test %*% b))
  )
  
  p_hat <- rowMeans(p_test)
  
  pred <- ifelse(p_hat > 0.5, 1, 0)
  acc_bayes[r] <- mean(pred == y_test)
  ll_bayes[r]  <- -mean(y_test * log(p_hat) +
                          (1 - y_test) * log(1 - p_hat))
  
  # ---- MLE baseline ----
  df_train <- data.frame(y = y_train, X_train)
  df_test  <- data.frame(y = y_test,  X_test)
  
  fit_mle <- glm(y ~ ., data = df_train, family = binomial())
  p_mle <- predict(fit_mle, newdata = df_test, type = "response")
  
  pred_mle <- ifelse(p_mle > 0.5, 1, 0)
  acc_mle[r] <- mean(pred_mle == y_test)
  ll_mle[r]  <- -mean(y_test * log(p_mle) +
                        (1 - y_test) * log(1 - p_mle))
}


# 汇总结果


cat("\n=== Repeated Random Split Results (10 runs) ===\n")

cat("\nBayesian Logistic Regression:\n")
cat("Accuracy: ",
    round(mean(acc_bayes),4), "±", round(sd(acc_bayes),4), "\n")
cat("LogLoss: ",
    round(mean(ll_bayes),4), "±", round(sd(ll_bayes),4), "\n")

cat("\nMLE Logistic Regression:\n")
cat("Accuracy: ",
    round(mean(acc_mle),4), "±", round(sd(acc_mle),4), "\n")
cat("LogLoss: ",
    round(mean(ll_mle),4), "±", round(sd(ll_mle),4), "\n")


# Plot control switch（可选）
PLOT <- TRUE

if (PLOT) {
  
  
  # 1. Trace plots（MCMC 收敛诊断）
 
  par(mfrow = c(2,2))
  
  plot(beta_post[,1], type="l",
       main="Trace plot: beta (npreg)",
       xlab="Iteration", ylab="Value")
  
  plot(beta_post[,2], type="l",
       main="Trace plot: beta (glu)",
       xlab="Iteration", ylab="Value")
  
  plot(beta_post[,5], type="l",
       main="Trace plot: beta (bmi)",
       xlab="Iteration", ylab="Value")
  
  plot(sigma2_post, type="l",
       main="Trace plot: sigma^2",
       xlab="Iteration", ylab="Value")
  
  
  # 2. ACF plots（混合性诊断）
  par(mfrow = c(2,2))
  
  acf(beta_post[,1], main="ACF: beta (npreg)")
  acf(beta_post[,2], main="ACF: beta (glu)")
  acf(beta_post[,5], main="ACF: beta (bmi)")
  acf(sigma2_post, main="ACF: sigma^2")
  
  

  # 3. Posterior density plots
  par(mfrow = c(2,2))
  
  dens <- density(beta_post[,1])
  plot(dens, main="Posterior density: beta (npreg)")
  abline(v=0, lty=2)
  
  dens <- density(beta_post[,2])
  plot(dens, main="Posterior density: beta (glu)")
  abline(v=0, lty=2)
  
  dens <- density(beta_post[,5])
  plot(dens, main="Posterior density: beta (bmi)")
  abline(v=0, lty=2)
  
  dens <- density(beta_post[,7])
  plot(dens, main="Posterior density: beta (age)")
  abline(v=0, lty=2)
  
  
  # 4. Posterior predictive distribution（测试集示例）

  # 选一个测试样本
  i <- 1
  p_post_single <- p_test[,i]
  
  hist(p_post_single, breaks=30, probability=TRUE,
       main="Posterior predictive distribution",
       xlab="Predicted probability")
  
  abline(v = mean(p_post_single), lwd=2)
  abline(v = quantile(p_post_single, c(0.025,0.975)), lty=2)
  
  
 
  # 5. Bayesian vs MLE 系数对比
  beta_mle <- coef(fit_mle)[-1]
  
  plot(beta_mle, beta_mean,
       xlab="MLE estimates",
       ylab="Bayesian posterior mean",
       main="Bayesian vs MLE coefficients")
  
  abline(0,1,lty=2)
  text(beta_mle, beta_mean,
       labels=names(beta_mle), pos=4)
}
