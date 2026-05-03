functions {
  real lda_partial_sum(array[] int count_slice,
                       int start, int end,
                       array[] int d,
                       array[] int w,
                       array[] vector theta,
                       array[] vector phi,
                       int K) {
    real pt = 0;
    for (i in 1:(end - start + 1)) {
      int n = start + i - 1;
      vector[K] gamma;
      for (k in 1:K) {
        gamma[k] = log(theta[d[n], k]) + log(phi[k, w[n]]);
      }
      pt += count_slice[i] * log_sum_exp(gamma);
    }
    return pt;
  }
}
data {
  int<lower=2> K;               
  int<lower=2> V;               
  int<lower=1> M;               
  int<lower=1> N_unique;        
  array[N_unique] int<lower=1,upper=M> d;
  array[N_unique] int<lower=1,upper=V> w;
  array[N_unique] int<lower=1> count;
  vector<lower=0>[K] alpha;     
  array[K] simplex[V] phi;
}
parameters {
  array[M] simplex[K] theta;
}
model {
  for (m in 1:M) theta[m] ~ dirichlet(alpha);
  
  int grainsize = 256;
  target += reduce_sum(lda_partial_sum, count, grainsize, d, w, theta, phi, K);
}