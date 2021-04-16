/*
 * particle_filters.c
 *
 *  Created on: 18 Sep 2020
 *      Author: heine
 */
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#include "particle_filters.h"

void MLbootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
                       int* sample_sizes, double *worst_case_sign_ratio, double* x_hats, double *filtering_time, int* mesh_sizes,  int kl, int ku, double* ab0, double* ab1, int ldab, int* ipiv0, int* ipiv1, int ldb0, int ldb1, int info, double w, double d, double E, double I, double h0, double h1, double L , double* meass, int N_meas) {
  
  int make_correction = 1; // If one ML telescoping correction is made, if 0 not.
  int N_levels = 2; // hard coded for now
  int N = 0;
  
  const char prog_bar[51] = "-------------------------------------------------";
  int* minds0 = (int*)malloc(2 * N_meas * sizeof(int));
  int* minds1 = minds0 + N_meas;
  double* obss = (double*)malloc(N_meas*sizeof(double));
  double lower_bound = 0.25, upper_bound = L - 0.25;
  
  clock_t start = clock();
  int bar_segment_count = 0;
  
  for (short i = 0; i < N_levels; i++) {
    N += sample_sizes[i];
    printf("N%i = %i, ", i, sample_sizes[i]);
  }
  printf("N = %i, L0 = %i, L1 = %i\n", N, mesh_sizes[0],mesh_sizes[1]);
  
  double *X = (double*) malloc(2 * N * sizeof(double));
  double W1_tmp;
  double* B0 = (double*) malloc(sample_sizes[0]*mesh_sizes[0]*sizeof(double));
  double* B1 = (double*) malloc(sample_sizes[1]*mesh_sizes[1]*sizeof(double));
  double* pred_meass0 = (double*) malloc(2 * sample_sizes[1] * N_meas * sizeof(double));
  double* pred_meass1 = pred_meass0 + sample_sizes[1] * N_meas;
  double* W = (double*) malloc(2 * N * sizeof(double));
  double* absW = W + N;
  short* signs = (short*) malloc(2 * N * sizeof(short));
  short* signs_res = signs + N;
  int* ind = (int*) malloc(2 * N * sizeof(int));
  int* permutation = ind + N; // needed for uniform resampling
  
  double x_hat = 0;
  double p_hat = 0;
  
  double* X0 = X;
  double* X1 = X + sample_sizes[0];
  double* W0 = W;
  double* W1 = W + sample_sizes[0];
  double* X_res = X + N;
  
  printf(
         "%s\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
         prog_bar);
  fflush(stdout);
  
  /* Make copies of the band matrices as DGBSV destroys them */
  double* a0_copy = (double*) malloc(ldab * mesh_sizes[0] * sizeof(double));
  double* a1_copy = (double*) malloc(ldab * mesh_sizes[1] * sizeof(double));
  
  /*
   * Initial sample and level indicator
   */
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, clock());
  for (int i = 0; i < N; i++) {
    X[i] = gsl_ran_gaussian(rng, sig_std) + x0; // init sample
    W[i] = (double) 1.0 / (double) N; // init weights
    signs_res[i] = 1;
  }
  
  double normaliser = 0;
  double abs_normaliser = 0;
  //  double diff;
  double ESS, ESSABS;
  //  double tmp_x_hat;
  double positive_mass, negative_mass;
  
  //  FILE* bpf_out = fopen("mlbpf_out.txt", "w");
  
  
  clock_t iteration_timer;
  clock_t resampling_timer;
  clock_t level_timer_start;
  double level_0_particle_cost = 0;
  double level_1_particle_cost = 0;
  double rsamp_time_per_prcl = 0;
  worst_case_sign_ratio[0] = 100;
  double sign_balance;
  double obs_std_scale = 0.8;
  double intercepts[2];
  double slopes[2];
    
  int res_pos, res_neg;
  int dodb = 0;
  
  // Mesh point indices for the measurement locations
  for(int i = 0; i < N_meas; i++) {
    minds0[i] = (int)(meass[i] / L * (double) mesh_sizes[0]);
    minds1[i] = (int)(meass[i] / L * (double) mesh_sizes[1]);
  }
  
  /*
   * FILTER MAIN LOOP
   */
  for (int n = 0; n < data_length; n++) {
      
    // Restore the destroyed band matrix
    array_copy(ab0, a0_copy, ldab * mesh_sizes[0]);
    array_copy(ab1, a1_copy, ldab * mesh_sizes[1]);
    
    iteration_timer = clock(); // start the stopwatch
    level_timer_start = clock();
    
    // Extract the observations for the current time step
    for(int i = 0; i < N_meas; i++)
      obss[i] = y[n + i * data_length];
    
    /*
     * Level 1 weight calculation
     */
    
    /* Calculate Level 1 first, so we can estimate the bias of the observation mapping */
    
    /* Solve the beam for level 0 mesh and sample N1 */
    ebb_solve(sample_sizes[1], X1, mesh_sizes[0], B1, w, d, h0, E, I, L, kl, ku, a0_copy, ldab, ipiv0, ldb0, info, N_meas, minds0, pred_meass0);
    
    /* Solve the beam for level 1 mesh and sample N1 */
    ebb_solve(sample_sizes[1], X1, mesh_sizes[1], B1, w, d, h1, E, I, L, kl, ku, a1_copy, ldab, ipiv1, ldb1, info, N_meas, minds1, pred_meass1);
    
    /* Fit a first order LS model for the error between level 0 and level 1 */
    if(sample_sizes[1] > 0){ // do only if N1 > 0
      ls_fit(N_meas, sample_sizes[1], X1, pred_meass0, pred_meass1, intercepts, slopes);
    }
    
    /* Now that we know the bias, we can evaluate the weights for level 1 */
    double diff;
    for (int j = 0; j < sample_sizes[1]; j++) {
      W1[j] = 1;
      W1_tmp = 1;
      for(int k = 0; k < N_meas; k++){
        /* Level 0 */
        diff = obss[k] - pred_meass0[sample_sizes[1] * k + j] - intercepts[k] - slopes[k] * X1[j];
        W1_tmp *= exp( - diff * diff / (double) 2.0 / (obs_std * obs_std_scale) / (obs_std * obs_std_scale));
        /* Level 1 */
        diff = obss[k] - pred_meass1[sample_sizes[1] * k + j];
        W1[j] *= exp( - diff * diff / (double) 2.0 / obs_std / obs_std );
      }
      
      W1[j] -= W1_tmp; // take the difference and...
      W1[j] /= (double) sample_sizes[1]; // ... normalise by sample size N1
//      W1[j] = 0; /* uncomment to disable level 1 correction*/
    }
    
    /*
     Level 0 weight calculation
     */
    
    array_copy(ab0, a0_copy, ldab * mesh_sizes[0]);
    
    likelihood(X0, sample_sizes[0], obss, obs_std_scale*obs_std, mesh_sizes[0], kl, ku, a0_copy, ldab, ipiv0, ldb0, info, B0, w, d, h0, E, I, W0, L, minds0, N_meas, dodb, intercepts, slopes);
    
    // normalise by the sample size (N0)
    for (int i = 0; i < sample_sizes[0]; i++)
      W[i] /= (double)sample_sizes[0];
    
    if(!make_correction) {
      for(int i = 0; i < sample_sizes[1]; i++) {
        W1[i] = 0;
      }
    }
    
    /* Joint normalisation of the weights */
    normaliser = 0;
    abs_normaliser = 0;
    for (long i = 0; i < N; i++) {
      
      W[i] *= (double) signs_res[i]; // Take the sign into account
      
      absW[i] = fabs(W[i]);
      signs[i] = W[i] > 0 ? 1 : -1;
      normaliser += W[i];
      abs_normaliser += absW[i];
    }
    
    /* Normalise */
    positive_mass = 0;
    negative_mass = 0;
    for (long i = 0; i < N; i++) {
      W[i] /= normaliser;
      if (W[i] > 0) {
        positive_mass += W[i];
      } else {
        negative_mass += fabs(W[i]);
      }
      absW[i] /= abs_normaliser;
    }
    sign_balance = positive_mass / negative_mass;
    if (sign_balance < worst_case_sign_ratio[0]) {
      worst_case_sign_ratio[0] = sign_balance;
    }
    
    ESS = 0;
    ESSABS = 0;
    for (long i = 0; i < N; i++) {
      ESS += W[i] * W[i];
      ESSABS += absW[i] * absW[i];
    }
    ESS = (double) 1.0 / ESS;
    ESSABS = (double) 1.0 / ESSABS;
    
    /* Resample */
    resampling_timer = clock();
    resample(N, absW, ind, rng);
    random_permuter(permutation, N, rng);
    rsamp_time_per_prcl += (double) (clock() - resampling_timer) / CLOCKS_PER_SEC / (double) N;
    
    normaliser = 0;
    res_pos = 0;
    res_neg = 0;
    for (long i = 0; i < N; i++) {
      X_res[permutation[i]] = X[ind[i]];
      signs_res[permutation[i]] = signs[ind[i]];
      normaliser += signs[ind[i]];
      if(signs[ind[i]]>0) {
        res_pos++;
      }else{
        res_neg++;
      }
    }
    for (long i = 0; i < N; i++) {
      W[i] = (double) signs_res[i] / normaliser;
    }
    
    /* Calculate the output: posterior mean */
    if(x_hats!=NULL){
      x_hat = 0;
      for (int i = 0; i < N; i++) {
        x_hat += X_res[i] * W[i];
        if(isnan(x_hat)){
          printf("NaN detected!\n X_res[i] = %e, W[i] = %e, i = %i, normaliser = %e + = %e, - = %e\nESS=%e, res_+ =%i, res_- = %i\n",
                 X_res[i], W[i], i,normaliser,positive_mass, negative_mass,ESS,res_pos,res_neg);
          fflush(stdout);
          getchar();
        }
      }
      x_hats[n] = x_hat;
      
      /* Calculate the output: posterior variance */
      p_hat = 0;
      for (long i = 0; i < N; i++) {
        diff = X_res[i] - x_hat;
        p_hat += diff * diff * W[i];
      }
    }
    
    /* Mutate */
    for (long i = 0; i < N; i++) {
      X[i] = X_res[i] + gsl_ran_gaussian(rng, sig_std);
      
      if(X[i] < lower_bound)
        X[i] = lower_bound + lower_bound-X[i];
      if(X[i] > upper_bound)
        X[i] = upper_bound - (X[i]-upper_bound);
    }
    
    if (floor(n * (double) 50 / data_length) > bar_segment_count) {
      printf("█");
      fflush(stdout);
      bar_segment_count++;
    }
    
  }
  
  filtering_time[0] = (double) (clock() - start) / CLOCKS_PER_SEC / (double) data_length;
  filtering_time[1] = level_0_particle_cost / (double) data_length / (double) sample_sizes[0];
  filtering_time[2] = level_1_particle_cost / (double) data_length / (double) sample_sizes[1];
  filtering_time[3] = rsamp_time_per_prcl / (double) data_length / (double) N;
  printf(" %5.5f sec\n", filtering_time[0]);
  
  free(X);
  free(W);
  free(signs);
  free(ind);
  free(a0_copy);
  free(a1_copy);
  free(B0);
  free(B1);
  free(minds0);
  free(obss);
  free(pred_meass0);
  gsl_rng_free(rng);
  
}

void bootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
                     int N, double* x_hats, double* p_hats, double *filtering_time, int mesh_size,	int kl, int ku, double* ab, int ldab, int* ipiv, int ldb, int info, double w, double d, double E, double I, double h , double L, double *meass, int N_meas) {
  
  const char prog_bar[51] = "-------------------------------------------------";
  clock_t start;
  int bar_segment_count = 0;
  
  printf("Basic BPF with N = %i\n", N);
  
  double *X = (double*) malloc(2 * N * sizeof(double));
  double *X_res = X + N;
  double *W = (double*) malloc(N * sizeof(double));
  int *ind = (int*) malloc(N * sizeof(int));
  double* payloads = (double*) malloc(mesh_size * N * sizeof(double));
  // Because DGBSV destroys the AB matrix, we need to make a copy of it
  double* ab_copy = (double*) malloc(ldab * mesh_size * sizeof(double));
  double x_hat = 0;
  double p_hat = 0;
  double lower_bound = 0.25, upper_bound = L - 0.25;
  double intercepts[2] = {0,0};
  double slopes[2] = {0,0};
  
  printf(
         "%s\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
         prog_bar);
  fflush(stdout);
  
  
  start = clock();
  
  /*
   * Initial sample and level indicator
   */
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, clock());
  for (int i = 0; i < N; i++) {
    X[i] = gsl_ran_gaussian(rng, sig_std) + x0;
    W[i] = (double) 1.0 / (double) N;
  }
  
  double normaliser = 0;
  double diff;
  double ESS;
  double cost_per_particle = 0;
  
  int* minds = (int*)malloc(N_meas * sizeof(int));
  double* obss = (double*)malloc(N_meas * sizeof(double));
  for(int i = 0; i < N_meas; i++){
    minds[i] = (int)(meass[i] / L * (double) mesh_size);
  }
  
  /* Classic BPF main loop */
  /* --------------------- */
  for (int n = 0; n < data_length; n++) {
    
    /* Weight calculation */
    normaliser = 0;
    
    array_copy(ab, ab_copy, ldab * mesh_size);
    
    /*
     * Weight calculation
     */
    clock_t stopwatch_start = clock();
    for(int i = 0; i < N_meas; i++)
      obss[i] = y[n + i*data_length];
    
    likelihood(X, N, obss, obs_std, mesh_size, kl, ku, ab_copy, ldab, ipiv, ldb, info, payloads, w, d, h, E, I, W, L, minds, N_meas, 0, intercepts, slopes);
    clock_t stopwatch_stop = clock();
    
    cost_per_particle += (double)(stopwatch_stop-stopwatch_start)/(double) CLOCKS_PER_SEC;
    
    // Normalise the weights
    normaliser = 0;
    for (int i = 0; i < N; i++)
      normaliser += W[i];
    for (int i = 0; i < N; i++)
      W[i] /= normaliser;
    
    ESS = 0;
    for (int i = 0; i < N; i++) {
      ESS += W[i] * W[i];
    }
    ESS = (double) 1.0 / ESS;
    
    /* Resample */
    resample(N, W, ind, rng);
    for (int i = 0; i < N; i++) {
      X_res[i] = X[ind[i]];
    }
    
    /* Calculate the output: posterior mean */
    if(x_hats != NULL) {
      x_hat = 0;
      for (int i = 0; i < N; i++) {
        x_hat += X_res[i] / (double) N;
      }
      x_hats[n] = x_hat;
      /* Calculate the output: posterior variance */
      p_hat = 0;
      for (int i = 0; i < N; i++) {
        diff = X_res[i] - x_hat;
        p_hat += diff * diff / (double) (N-1);
      }
      if(p_hats!=NULL)
        p_hats[n] = p_hat;
    }
    /* Mutate */
    for (int i = 0; i < N; i++) {
      X[i] = X_res[i] + gsl_ran_gaussian(rng, sig_std);
      
      if(X[i] < lower_bound)
        X[i] = lower_bound + lower_bound-X[i];
      if(X[i] > upper_bound)
        X[i] = upper_bound - (X[i]-upper_bound);
    }
    
    if (floor(n * (double) 50 / data_length) > bar_segment_count) {
      printf("█");
      fflush(stdout);
      bar_segment_count++;
    }
  }
  filtering_time[0] = (double) (clock() - start) / CLOCKS_PER_SEC / (double) data_length;
  filtering_time[1] = cost_per_particle / (double) data_length;
  filtering_time[2] = 0;
  filtering_time[3] = 0;
  printf(" %5.5f sec\n", filtering_time[0]);
  
  free(X);
  free(W);
  free(ind);
  free(ab_copy);
  free(payloads);
  free(minds);
  free(obss);
  gsl_rng_free(rng);
}

/* Computes a first order LS fit for error between level 0 and level 1 */
void ls_fit(int N_meas, int N1, double* X1, double* pred_meass0, double* pred_meass1, double* intercepts, double* slopes) {
  
  double a,b,c,d,e,f;
  
  for(int j = 0; j < N_meas; j++) {
    
    /* First order polynomial fit to the bias */
    a = 0;
    b = 0;
    d = 0;
    e = 0;
    f = 0;
    for(int i = 0; i < N1; i++) {
      a++;
      b += X1[i];
      d += X1[i] * X1[i];
      e += pred_meass1[j * N1 + i] - pred_meass0[j * N1 + i];
      f += X1[i] * (pred_meass1[j * N1 + i] - pred_meass0[j * N1 + i]);
    }
    c = b;
    intercepts[j] = (e * d - b * f) / (a * d - b * c);
    slopes[j] = (-c * e  + a * f) / (a * d - b * c);
  }
}

void likelihood(double* x, int N, double* y, double std, int mesh_size, int kl, int ku, double* ab, int ldab, int* ipiv, int ldb, int info, double* b, double w, double d, double h, double E, double I, double* likes, double L, int* minds, int N_meas, int dodb, double* intercepts, double* slopes) {
  
  // Create the instantaneous payload
  create_payload(N, mesh_size, b, w, d, x, h, E, I, L);
  
  // Solve the Euler-Bernoulli beam for each particle
  dgbsv_(&mesh_size, &kl, &ku, &N, ab, &ldab, ipiv, b, &ldb, &info);
  
  double diff;
  for (int j = 0; j < N; j++) {
    diff = 1;
    likes[j] = 1;
    for(int k = 0; k < N_meas; k++){
      diff = y[k] - b[j * mesh_size + minds[k]] - x[j]*slopes[k] - intercepts[k];
      likes[j] *= exp( - diff * diff / (double) 2.0 / std / std) ;
    }
  }
}

void random_permuter(int *permutation, int N, gsl_rng *r) {
  
  for (int i = 0; i < N; i++)
    permutation[i] = i;
  
  int j;
  int tmp;
  for (int i = N - 1; i > 0; i--) {
    j = (int)gsl_rng_uniform_int(r, i + 1);
    tmp = permutation[j];
    permutation[j] = permutation[i];
    permutation[i] = tmp;
  }
  
}

void resample(int size, double *w, int *ind, gsl_rng *r) {
  
  /* Generate the exponentials */
  double *e = (double*) malloc((size + 1) * sizeof(double));
  double g = 0;
  for (int i = 0; i <= size; i++) {
    e[i] = gsl_ran_exponential(r, 1.0);
    g += e[i];
  }
  /* Generate the uniform order statistics */
  double *u = (double *) malloc((size + 1) * sizeof(double));
  u[0] = 0;
  for (int i = 1; i <= size; i++)
    u[i] = u[i - 1] + e[i - 1] / g;
  
  /* Do the actual sampling with inverse cdf */
  double cdf = w[0];
  int j = 0;
  for (int i = 0; i < size; i++) {
    while (cdf < u[i + 1]) {
      j++;
      cdf += w[j];
    }
    ind[i] = j;
  }
  
  free(e);
  free(u);
}

void create_payload(int N, int n, double *b, double w, double d, double* x, double h, double E, double I, double L) {
  
  double g = -9.81; // Gravity acceleration
  double scaler = h * h * h * h / (E * I);
  double slab_thickness = 0.20; // meters
  double density = 7874; // kg/ m^3
  double slab_width = 0.72; // m
  int location_index_centre = 0, location_index_right = 0, location_index_left = 0;
  double deviation = slab_thickness / (double) 8;
  double diff;
  
  for (long j = 0; j < N;j++) { // iterate over particles
    
    for(int i = 0; i < n; i++)
      b[j * n + i] = g * w * d * (double)480;
    
    // Find the index corresponding to the location x
    location_index_centre = (int)(x[j] / L * (double) n);
    
    // Left end of the slab
    location_index_left = location_index_centre -  ceil(slab_thickness / h / (double) 2);
    // make sure index is non-negative
    location_index_left = location_index_left < 0 ? 0 : location_index_left;
    
    // Right end of the slab
    location_index_right = location_index_centre +  ceil(slab_thickness / h / (double) 2);
    // make sure index is less than mesh size
    location_index_right = location_index_right > n - 1 ? n - 1 : location_index_right;
    
    if(location_index_centre < 0) {
      printf("NEGATIVE INDEX!\n");
      fflush(stdout);
    }
    
    // Add weight to the location
    for(int i = location_index_left; i <= location_index_right ; i++){
      diff = x[j] - i*h;
      
      b[j * n + i] -=  slab_width * slab_width * density * 9.81 * exp(- diff * diff / (double) 2 / deviation / deviation);
    }
    // Scale appropriately (there is some physical justification to this)
    for(int i = 0; i < n;i++) {
      b[j * n + i] *= scaler;
    }
  }
}

void ebb_solve(int N, double* x, int mesh_size, double* b, double w, double d, double h, double E, double I, double L, int kl, int ku, double *ab, int ldab, int *ipiv, int ldb, int info, int N_meas, int* minds, double* pred_measurements) {

  create_payload(N, mesh_size, b, w, d, x, h, E, I, L);
  dgbsv_(&mesh_size, &kl, &ku, &N, ab, &ldab, ipiv, b, &ldb, &info);

  for (int j = 0; j < N; j++) {
      for(int k = 0; k < N_meas; k++){
        pred_measurements[N * k + j] = b[j * mesh_size + minds[k]];
      }
    }
}

void array_copy(double *src, double* dst, int length) {
  for(int i = 0; i < length; i++)
    dst[i] = src[i];
}
