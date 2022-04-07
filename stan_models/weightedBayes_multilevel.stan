functions {
  real weight_f(real L_raw, real w_raw) {
    real L;
    real w;
    L = exp(L_raw);
    w = 0.5 + inv_logit(w_raw)/2;
    return log((w * L + 1 - w) ./ ((1 - w) * L + w));
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
  real weight1M;
  real weight2M;
  real<lower=0> sigma;
  vector<lower=0>[2] tau;
  matrix[2, agents] z_IDs;
  cholesky_factor_corr[2] L_u;
}

transformed parameters {
  matrix[agents, 2] IDs;
  IDs = (diag_pre_multiply(tau, L_u) * z_IDs)';
}

model {
  target += normal_lpdf(weight1M | 0, 1);
  target += normal_lpdf(weight2M | 0, 1);
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


// make the weights interpretable
// 



/*
generated quantities{
  array[trials] real log_lik;
  real<lower=0> sigma_prior;
  real prior_preds;
  real post_preds;
  
  sigma_prior = inv_logit(normal_rng(prior_mean, prior_sd));
  prior_preds = normal_lpdf(y | logit(Source1) +  logit(Source2), sigma_prior);
  post_preds = normal_lpdf(y | logit(Source1) +  logit(Source2), sigma);
  
  for (n in 1:trials) {
    log_lik[n] = normal_lpdf(choice[trials] | logit(source1[trial]) +  logit(source2[trial]), sigma);
  }
}

*/
