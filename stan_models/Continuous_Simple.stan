//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//


data {
  int<lower=0> N;
  vector[N] y;
  vector[N] Source1;
  vector[N] Source2;
}

parameters {
  real<lower=0> sigma;
}

model {
  target += normal_lpdf(sigma | 0.5, 0.25) -
      normal_lccdf(0 | 0.5, 0.25);
}

generated quantities{
  array[N] real log_lik;
  array[N] real prediction;
  
  for (n in 1:N){  
    log_lik[n] = normal_lpdf(y[n] | logit(Source1[n]) + logit(Source2[n]), sigma);
    
    prediction[n] = inv_logit(logit(Source1[n]) + logit(Source2[n]));
  }
  
}


