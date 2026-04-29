data {
  int<lower=2> K;               
  int<lower=2> V;               
  int<lower=1> M;               
  int<lower=1> N_unique;        
  array[N_unique] int<lower=1,upper=M> d;
  array[N_unique] int<lower=1,upper=V> w;
  array[N_unique] int<lower=1> count;
  vector<lower=0>[K] alpha;     
  array[K] simplex[V] phi;      // 关键区别：phi 现在是作为 Data 传入！
}
parameters {
  array[M] simplex[K] theta;    // 只采样 BEIR 的文档-主题分布
}
model {
  for (m in 1:M) theta[m] ~ dirichlet(alpha);
  
  for (n in 1:N_unique) {
    vector[K] gamma;
    for (k in 1:K) {
      gamma[k] = log(theta[d[n], k]) + log(phi[k, w[n]]);
    }
    target += count[n] * log_sum_exp(gamma);
  }
}