import json
import numpy as np
from cmdstanpy import CmdStanModel
import scipy.stats

# 1. 编译 Stan 模型
model_train = CmdStanModel(stan_file='lda_train.stan')
model_infer = CmdStanModel(stan_file='lda_infer.stan')

# 2. 训练 MS MARCO (In-Domain)
print("开始在 MS MARCO 上进行 MCMC 采样...")
with open("data_msmarco.json", "r") as f:
    ms_data = json.load(f)

fit_ms = model_train.sample(data=ms_data, chains=2, iter_warmup=500, iter_sampling=500, show_progress=True)

# 提取 phi (主题-词汇分布) 的后验均值供 BEIR 推断使用
phi_samples = fit_ms.stan_variable(var='phi') 
phi_mean = np.mean(phi_samples, axis=0).tolist() # 形状: [K, V]

# 提取并计算 MS MARCO 的不确定性
theta_ms_samples = fit_ms.stan_variable(var='theta') # 形状: [samples, M, K]
ms_ci_widths = np.percentile(theta_ms_samples, 97.5, axis=0) - np.percentile(theta_ms_samples, 2.5, axis=0)
ms_theta_mean = np.mean(theta_ms_samples, axis=0)
ms_entropy = scipy.stats.entropy(ms_theta_mean, axis=1, base=2)

print(f"[MS MARCO] 平均 95% 置信区间宽度: {np.mean(ms_ci_widths):.4f}")
print(f"[MS MARCO] 平均香农熵: {np.mean(ms_entropy):.4f}\n")

# 3. 推断 BEIR 数据集 (Out-of-Domain)
with open("data_beir.json", "r") as f:
    beir_data_dict = json.load(f)

for db_name, b_data in beir_data_dict.items():
    print(f"开始在 BEIR [{db_name}] 上进行 MCMC 推断...")
    # 补充模型需要的额外参数
    b_data["K"] = ms_data["K"]
    b_data["V"] = ms_data["V"]
    b_data["alpha"] = ms_data["alpha"]
    b_data["phi"] = phi_mean  # 注入 MS MARCO 训练出的 phi
    
    # 由于是纯推断，收敛更快，采样数可适当减少
    fit_beir = model_infer.sample(data=b_data, chains=2, iter_warmup=300, iter_sampling=300, show_progress=False)
    
    theta_b_samples = fit_beir.stan_variable(var='theta')
    b_ci_widths = np.percentile(theta_b_samples, 97.5, axis=0) - np.percentile(theta_b_samples, 2.5, axis=0)
    b_theta_mean = np.mean(theta_b_samples, axis=0)
    b_entropy = scipy.stats.entropy(b_theta_mean, axis=1, base=2)
    
    print(f"[{db_name}] 平均 95% 置信区间宽度: {np.mean(b_ci_widths):.4f}")
    print(f"[{db_name}] 平均香农熵: {np.mean(b_entropy):.4f}\n")