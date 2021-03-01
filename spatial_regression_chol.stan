data {
  int<lower=0> N; // number of observed data points 
  vector[N] y; // observed response
  matrix[N,N] dist; // observed distance matrix
  real phi_lower; // lower point for phi (range)
  real phi_upper; // upper point for phi (range)
  int<lower=0> N_preds; // number of predictive points
  matrix[N_preds,N_preds] dist_preds; // distance matrix for predictive points
  matrix[N, N_preds] dist_12; //distance between observed and predicted
  real phi_a;
  real phi_b;
  real sigmasq_a;
  real sigmasq_b;
  real tausq_a;
  real tausq_b;
}

parameters {
  real<lower = phi_lower, upper = phi_upper> phi;
  real<lower = 0> sigmasq;
  real<lower = 0> tausq;
  real mu;
}

transformed parameters{
  vector[N] mu_vec;
  vector[N] tausq_vec;
  corr_matrix[N] Sigma;
  
  for(i in 1:N) mu_vec[i] = mu;
  for(i in 1:(N-1)){
   for(j in (i+1):N){
     Sigma[i,j] = exp((-1)*dist[i,j]/ phi);
     Sigma[j,i] = Sigma[i,j];
   }
 }
 for(i in 1:N) Sigma[i,i] = 1;
 for(i in 1:N) tausq_vec[i] = tausq;
}

model {
  matrix[N, N] L;
  L = cholesky_decompose(sigmasq * Sigma + diag_matrix(tausq_vec));

  y ~ multi_normal_cholesky(mu_vec, L);
  phi ~ inv_gamma(phi_a, phi_b);
  sigmasq ~ inv_gamma(sigmasq_a, sigmasq_b);
  tausq ~ inv_gamma(tausq_a, tausq_b);
  mu ~ normal(0, 10);
}

generated quantities {
  vector[N_preds] y_preds;
  vector[N] y_diff;
  vector[N_preds] mu_preds;
  corr_matrix[N_preds] Sigma_preds;
  vector[N_preds] tausq_preds;
  matrix[N, N_preds] Sigma_12;

  for(i in 1:N_preds) tausq_preds[i] = tausq;
  for(i in 1:N_preds) mu_preds[i] = mu;
  for(i in 1:N) y_diff[i] = y[i] - mu;
  

  for(i in 1:(N_preds-1)){
   for(j in (i+1):N_preds){
     Sigma_preds[i,j] = exp((-1)*dist_preds[i,j]/ phi);
     Sigma_preds[j,i] = Sigma_preds[i,j];
   }
 }
 for(i in 1:N_preds) Sigma_preds[i,i] = 1;
 
   for(i in 1:(N)){
   for(j in (1):N_preds){
     Sigma_12[i,j] = exp((-1)*dist_12[i,j]/ phi);
   }
 }

 y_preds = multi_normal_rng(mu_preds + (sigmasq * Sigma_12)' * inverse(sigmasq * Sigma) * (y_diff), sigmasq * Sigma_preds + diag_matrix(tausq_preds) - (sigmasq * Sigma_12)' * inverse(sigmasq * Sigma + diag_matrix(tausq_vec)) * (sigmasq * Sigma_12) );
}
