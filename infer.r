library(rstan)
library(jsonlite)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

calculate_entropy <- function(theta_mean) {
  # 计算香农熵，忽略概率为0的情况防止log(0)
  apply(theta_mean, 1, function(p) {
    p <- p[p > 0]
    -sum(p * log2(p))
  })
}

# 1. 加载和编译模型
cat("编译 Stan 模型...\n")
model_train <- stan_model(file = "lda_train.stan")
model_infer <- stan_model(file = "lda_infer.stan")

# 2. 训练 MS MARCO (In-Domain)
cat("开始在 MS MARCO 上进行 MCMC 采样...\n")
ms_data <- fromJSON("data_msmarco.json")
fit_ms <- sampling(model_train, data = ms_data, chains = 2, iter = 1000, warmup = 500)

# 提取后验
phi_samples <- extract(fit_ms, pars = "phi")$phi
phi_mean <- apply(phi_samples, c(2, 3), mean) # 计算每一维的均值

theta_ms_samples <- extract(fit_ms, pars = "theta")$theta
ms_theta_lower <- apply(theta_ms_samples, c(2, 3), quantile, probs = 0.025)
ms_theta_upper <- apply(theta_ms_samples, c(2, 3), quantile, probs = 0.975)
ms_ci_widths <- ms_theta_upper - ms_theta_lower

ms_theta_mean <- apply(theta_ms_samples, c(2, 3), mean)
ms_entropy <- calculate_entropy(ms_theta_mean)

cat(sprintf("[MS MARCO] 平均 95%% 置信区间宽度: %.4f\n", mean(ms_ci_widths)))
cat(sprintf("[MS MARCO] 平均香农熵: %.4f\n\n", mean(ms_entropy)))

# 3. 推断 BEIR 数据集 (Out-of-Domain)
beir_data_dict <- fromJSON("data_beir.json", simplifyVector = FALSE)
db_names <- names(beir_data_dict)

for (db_name in db_names) {
  cat(sprintf("开始在 BEIR [%s] 上进行 MCMC 推断...\n", db_name))
  b_data <- beir_data_dict[[db_name]]
  
  # 补充参数，注意 R 中的类型转换
  b_data$K <- ms_data$K
  b_data$V <- ms_data$V
  b_data$alpha <- ms_data$alpha
  b_data$phi <- phi_mean 
  
  # 在 Rstan 中，如果 list 中的向量只有 1 个元素，可能会被降维，由于我们是数组，通常是安全的
  fit_beir <- sampling(model_infer, data = b_data, chains = 2, iter = 600, warmup = 300, show_messages = FALSE)
  
  theta_b_samples <- extract(fit_beir, pars = "theta")$theta
  b_theta_lower <- apply(theta_b_samples, c(2, 3), quantile, probs = 0.025)
  b_theta_upper <- apply(theta_b_samples, c(2, 3), quantile, probs = 0.975)
  b_ci_widths <- b_theta_upper - b_theta_lower
  
  b_theta_mean <- apply(theta_b_samples, c(2, 3), mean)
  b_entropy <- calculate_entropy(b_theta_mean)
  
  cat(sprintf("[%s] 平均 95%% 置信区间宽度: %.4f\n", db_name, mean(b_ci_widths)))
  cat(sprintf("[%s] 平均香农熵: %.4f\n\n", db_name, mean(b_entropy)))
}