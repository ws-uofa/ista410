data {
  int<lower=2> K;               // 主题数量
  int<lower=2> V;               // 词汇表大小
  int<lower=1> M;               // 文档数量
  int<lower=1> N_unique;        // 唯一 (文档,词汇) 对的数量
  array[N_unique] int<lower=1,upper=M> d;
  array[N_unique] int<lower=1,upper=V> w;
  array[N_unique] int<lower=1> count;
  vector<lower=0>[K] alpha;     // theta 先验
  vector<lower=0>[V] beta;      // phi 先验
}
parameters {
  array[M] simplex[K] theta;
  array[K] simplex[V] phi;
}
model {
  // 先验
  for (m in 1:M) theta[m] ~ dirichlet(alpha);
  for (k in 1:K) phi[k] ~ dirichlet(beta);
  
  // 似然 (使用 count 优化性能)
  for (n in 1:N_unique) {
    vector[K] gamma;
    for (k in 1:K) {
      gamma[k] = log(theta[d[n], k]) + log(phi[k, w[n]]);
    }
    target += count[n] * log_sum_exp(gamma);
  }
}