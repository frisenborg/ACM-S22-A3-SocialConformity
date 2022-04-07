
data {
  int<lower=0> N;
  int<lower=0> agents;
  array[N, agents] real y;
  array[N, agents] real Source1;
  array[N, agents] real Source2;
}

parameters {
  real<lower=0> sigma;
}

model {
  target += normal_lpdf(sigma | 0.5, 0.25) -
      normal_lccdf(0 | 0.5, 0.25);
}

generated quantities{
  array[N, agents] real log_lik;
  array[N, agents] real prediction;
  
  for (agent in 1:agents){
    for (n in 1:N){
      log_lik[n, agent] = normal_lpdf(y[n, agent] | logit(Source1[n, agent]) + logit(Source2[n, agent]), sigma);
      
      prediction[n, agent] = inv_logit(logit(Source1[n, agent]) + logit(Source2[n, agent]));
    }
  }
}


