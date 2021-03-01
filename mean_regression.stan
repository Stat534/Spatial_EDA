data {
  int<lower=0> N; // number of observed data points 
  vector[N] y; // observed response
  int<lower=0> N_preds; // number of predictive points
}

parameters {
  real<lower = 0> tausq;
  real mu;
}

transformed parameters{
  vector[N] mu_vec;
  vector[N] tausq_vec;
  for(i in 1:N) mu_vec[i] = mu;
  for(i in 1:N) tausq_vec[i] = tausq;
}

model {
  y ~ multi_normal(mu_vec ,diag_matrix(tausq_vec));
  mu ~ normal(0, 10);
  
}

generated quantities {
  vector[N_preds] y_preds;
  vector[N_preds] mu_preds;
  vector[N_preds] tausq_preds;

  for(i in 1:N_preds) mu_preds[i] = mu;
  for(i in 1:N_preds) tausq_preds[i] = tausq;

  y_preds = multi_normal_rng(mu_preds, diag_matrix(tausq_preds));
}
