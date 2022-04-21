data {
  int<lower=0> N;
  array[N] int y;
  vector[N] Source1;
  vector[N] Source2;
}

generated quantities{
  array[N] real log_lik;
  
  for (n in 1:N){  
    log_lik[n] = bernoulli_logit_lpmf(y[n] | logit(Source1[n]) + logit(Source2[n]));
  }
  
}


