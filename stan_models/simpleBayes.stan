
functions{
  // Create a function for random number generating a truncated normal
  real normal_trunc_rng(real mu, real sigma, real lower_boundary) {
    real p = normal_cdf(lower_boundary | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower=0> trials;
  int<lower=0> agents;
  array[trials, agents] real outcome;
  array[trials, agents] real source1;
  array[trials, agents] real source2;
}

parameters {
  real<lower=0> sigma;
}

model {
  target += normal_lpdf(sigma | 0.5, 0.25) -
      normal_lccdf(0 | 0.5, 0.25);
}

generated quantities{
  array[trials, agents] real log_lik;
  array[trials, agents] real prediction;
  real sigma_prior;
  
  // Loop through each agent and trial
  for (agent in 1:agents){
    for (trial in 1:trials){
      
      // We can calculate the log_likelihood of an outcome given the model
      log_lik[trial, agent] = normal_lpdf(
        outcome[trial, agent] | logit(source1[trial, agent]) + logit(source2[trial, agent]),
        sigma);
      
      // We can also predict an outcome from the model
      prediction[trial, agent] = inv_logit(logit(source1[trial, agent]) + logit(source2[trial, agent]));
    }
  }
  
  // Lastly, we generate predictions from the sigma prior
  sigma_prior = normal_trunc_rng(0.5, 0.25, 0);
}


