functions {
  real weight_f(real L_raw, real w_raw) {
    real L;
    real w;
    L = exp(L_raw);
    w = 0.5 + inv_logit(w_raw)/2;
    return log((w * L + 1 - w) ./ ((1 - w) * L + w));
  }
    
  real normal_lb_rng(real mu, real sigma, real lb) {
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

data {
  int<lower=1> trials;
  int<lower=1> agents;
  array[trials, agents] real choice;
  array[trials, agents] real source1;
  array[trials, agents] real source2;
}

parameters {
 // Mean of the distributions of weights for the two sources in the overall distributions.
  real weight1M;
  real weight2M;
  real<lower=0> sigma;  // overall sd of the outcome
  vector<lower=0>[2] tau;   // sd of the weights
  matrix[2, agents] z_IDs;  // z score of individual differences
  cholesky_factor_corr[2] L_u;  // correlation of the weights
}

transformed parameters {
  matrix[agents, 2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {
  target += normal_lpdf(weight1M | 0, 1);
  target += normal_lpdf(weight2M | 0, 1);
  target += normal_lpdf(sigma | 0.2, 0.1) -
      normal_lccdf(0 | 0.2, 0.1); // We should check this is best way to define sigma 
  target += normal_lpdf(tau[1] | 0, .3) -
    normal_lccdf(0 | 0, .3);
  target += normal_lpdf(tau[2] | 0, .3) -
    normal_lccdf(0 | 0, .3);
  target += lkj_corr_cholesky_lpdf(L_u | 2);
  
  target += std_normal_lpdf(to_vector(z_IDs));
  
  for (agent in 1:agents) {
    for (trial in 1:trials) {
      target += normal_lpdf(choice[trial, agent] |
        weight_f(source1[trial, agent], weight1M + IDs[agent, 1]) +
        weight_f(source2[trial, agent], weight2M + IDs[agent, 2]),
        sigma);
    }
  }
}

generated quantities{
  array[trials, agents] real log_lik;
  real weight1M_prior;
  real weight2M_prior;
  real sigma_prior;
  real tau1_prior;
  real tau2_prior;
  
  
  weight1M_prior = normal_rng(0, 1);
  weight2M_prior = normal_rng(0, 1);
  sigma_prior = normal_lb_rng(0.2, 0.1, 0);
  tau1_prior = normal_lb_rng(0, 0.3, 0); 
  tau2_prior = normal_lb_rng(0, 0.3, 0);
  
  for (agent in 1:agents){
    for (trial in 1:trials){
      log_lik[trial, agent] = normal_lpdf(choice[trial, agent] |
        weight_f(source1[trial, agent], weight1M + IDs[1, agent]) +
        weight_f(source2[trial, agent], weight2M + IDs[2, agent]),
        sigma);
    }
  }
}

