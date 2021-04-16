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

#include "particle_filters.h"

void MLbootstrapfilter(double* y, double sig_std, double obs_std, int data_length,
    double x0, int Nobs, short N_levels, long* sample_sizes, double* x_hats,
    double *filtering_time, double *Rinv0, double *Rinv1, double *worst_case_sign_ratio) {

  const char prog_bar[51] = "-------------------------------------------------";

  clock_t start;
  int bar_segment_count = 0;

  long N = 0;
  for (short i = 0; i < N_levels; i++) {
    N += sample_sizes[i];
//    printf("N%i = %lu, ", i, sample_sizes[i]);
  }
//  printf("N = %lu\n", N);

  double *X = (double*) malloc(N * sizeof(double));
  double *X_res = (double*) malloc(N * sizeof(double));
  double *W = (double*) malloc(N * sizeof(double));
  double *absW = (double*) malloc(N * sizeof(double));
  short *signs = (short*) malloc(N * sizeof(short));
  short *signs_res = (short*) malloc(N * sizeof(short));
  long *ind = (long*) malloc(N * sizeof(long));
  short *level_indicator = (short*) malloc(N * sizeof(short));
  double x_hat = 0;
  double p_hat = 0;

  printf(
      "%s\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
      prog_bar);
  fflush(stdout);

  start = clock();

  /*
   * Initial sample and level indicator
   */
  long sample_counter = 0;
  short current_level = 0;
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng, clock());
  for (long i = 0; i < N; i++) {

    X[i] = gsl_ran_gaussian(rng, sig_std) + x0; // init sample
    W[i] = (double) 1.0 / (double) N; // init weights

    /* Construct level indicator array */
    signs_res[i] = 1;
    if (sample_counter >= sample_sizes[current_level]) {
      current_level++;
      sample_counter = 0;
    } else {
      sample_counter++;
    }
    level_indicator[i] = current_level;
  }

  double normaliser = 0;
  double abs_normaliser = 0;
  double diff;
  double ESS, ESSABS;
  double tmp_x_hat;
  double positive_mass, negative_mass;

  FILE* bpf_out = fopen("bpf_out.txt", "w");

  double* raw_like = (double*) malloc(2 * N * sizeof(double));
  long *permutation = (long*) malloc(N * sizeof(long));

  double lap1 = 0, lap2 = 0, lap3 = 0, lap4 = 0;
  clock_t iteration_timer;
  clock_t resampling_timer;
  clock_t level_timer_start;
  double level_0_particle_cost = 0;
  double level_1_particle_cost = 0;
  double rsamp_time_per_prcl = 0;
  double time_increment;
  double scaler;
  double num, den;
  worst_case_sign_ratio[0] = 100;
  double sign_balance;

  /* FILTER MAIN LOOP */
  for (int n = 0; n < data_length; n++) {

    iteration_timer = clock(); // start the stopwatch
    level_timer_start = clock();
    /*
     * Weight calculation
     */
    normaliser = 0;
    abs_normaliser = 0;
    for (long i = 0; i < N; i++) {
      if (level_indicator[i] == 0) {
        /* Use the diagonal observation covariance */
        raw_like[i] = likelihood(n, X[i], y, 0, Rinv0, Nobs, 1);
        raw_like[N + i] = 0;
      } else {
        /* Calculate both, the diagonal and full covariance to get level 1
         * differences */
        raw_like[i] = full_likelihood(n, X[i], y, Rinv1, Nobs, 1);
        raw_like[N + i] = likelihood(n, X[i], y, 0, Rinv0, Nobs, 1);
      }
      if (i == sample_sizes[0]) {
        /* When level indicator changes take the time per particle for level 0 */
        time_increment = (double) (clock() - level_timer_start) / CLOCKS_PER_SEC
            / (double) sample_sizes[0];
        level_0_particle_cost += time_increment;
        level_timer_start = clock();
      }
    }
    /* Time per particle for level 1 */
    time_increment = (double) (clock() - level_timer_start) / CLOCKS_PER_SEC
        / (double) sample_sizes[1];
    level_1_particle_cost += time_increment;
    lap1 += (double) (clock() - iteration_timer) / CLOCKS_PER_SEC;
    
    /*
     * Find scaler for the level 0 likelihood
     * */
    if (N_levels == 2) { // Do this only for non-unilevel
      num = 0;
      den = 0;
      for (int i = (int)sample_sizes[0]; i < N; i++) {
        num += raw_like[i] * raw_like[N + i];
        den += raw_like[N + i] * raw_like[N + i];
      }
      scaler = num / den;
      scaler = isnan(scaler) ? 1 : scaler;
      /*
       * Scale all level 0 weights by 'scaler'
       * */
      for (long i = 0; i < sample_sizes[0]; i++) // level 0
        raw_like[i] *= scaler;

      for (long i = sample_sizes[0]; i < N; i++) // Level 1
        raw_like[N + i] *= scaler;
    }

    /* Actual weight calculation */
    for (long i = 0; i < N; i++) {

      W[i] = (raw_like[i] - raw_like[N + i]) / (double) sample_sizes[level_indicator[i]]
          * (double) signs_res[i];

      absW[i] = fabs(W[i]);
      signs[i] = W[i] > 0 ? 1 : -1;
      normaliser += W[i];
      abs_normaliser += absW[i];
    }

    /* Normalise */
    tmp_x_hat = 0;
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
      tmp_x_hat += X[i] * W[i];
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

    lap2 += (double) (clock() - iteration_timer) / CLOCKS_PER_SEC;

    /* Resample */
    resampling_timer = clock();
    resample(N, absW, ind, rng);
    random_permuter(permutation, N, rng);
    rsamp_time_per_prcl += (double) (clock() - resampling_timer) / CLOCKS_PER_SEC / (double) N;

    lap3 += (double) (clock() - iteration_timer) / CLOCKS_PER_SEC;

    normaliser = 0;
    for (long i = 0; i < N; i++) {
      X_res[permutation[i]] = X[ind[i]];
      signs_res[permutation[i]] = signs[ind[i]];
      normaliser += signs[ind[i]];
    }
    for (long i = 0; i < N; i++) {
      W[i] = (double) signs_res[i] / normaliser;
    }

    /* Calculate the output: posterior mean */
    x_hat = 0;
    for (long i = 0; i < N; i++) {
      x_hat += X_res[i] * W[i];
    }
    x_hats[n] = x_hat;

    /* Calculate the output: posterior variance */
    p_hat = 0;
    for (long i = 0; i < N; i++) {
      diff = X_res[i] - x_hat;
      p_hat += diff * diff * W[i];
    }

    /* Mutate */
    for (long i = 0; i < N; i++)
      X[i] = X_res[i] + gsl_ran_gaussian(rng, sig_std);

    if (floor(n * (double) 50 / data_length) > bar_segment_count) {
      printf("█");
      fflush(stdout);
      bar_segment_count++;
    }
    lap4 += (double) (clock() - iteration_timer) / CLOCKS_PER_SEC;

  }

  filtering_time[0] = (double) (clock() - start) / CLOCKS_PER_SEC / (double) data_length;
  filtering_time[1] = level_0_particle_cost / (double) data_length;
  filtering_time[2] = level_1_particle_cost / (double) data_length;
  filtering_time[3] = rsamp_time_per_prcl / (double) data_length;
  printf(" %5.2f sec\n", filtering_time[0]);
  printf(" %5.2f  %5.2f  %5.2f  %5.2f\n", lap1, lap2, lap3, lap4);

  fclose(bpf_out);

  free(X);
  free(X_res);
  free(W);
  free(absW);
  free(signs);
  free(signs_res);
  free(ind);
  free(level_indicator);
  free(raw_like);
  free(permutation);
  gsl_rng_free(rng);

}

void bootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
                     int Nobs, long N, double* x_hats, double *filtering_time, double *Rinv, short full) {

  const char prog_bar[51] = "-------------------------------------------------";
  clock_t start;
  int bar_segment_count = 0;

  printf("Basic BPF with N = %lu\n", N);

  double *X = (double*) malloc(N * sizeof(double));
  double *X_res = (double*) malloc(N * sizeof(double));
  double *W = (double*) malloc(N * sizeof(double));
  long *ind = (long*) malloc(N * sizeof(long));
  double x_hat = 0;
  double p_hat = 0;

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
  for (long i = 0; i < N; i++) {
    X[i] = gsl_ran_gaussian(rng, sig_std) + x0;
    W[i] = (double) 1.0 / (double) N;
  }

  double normaliser = 0;
  double diff;
  double ESS;

  FILE* bpf_out = fopen("classicbpf_out.txt", "w");

  /* Classic BPF main loop */
  for (int n = 0; n < data_length; n++) {

    /* Weight calculation */
    normaliser = 0;

    /*
     * Weight calculation
     */
    for (long i = 0; i < N; i++) {
      if (full > 0) {
        W[i] = full_likelihood(n, X[i], y, Rinv, Nobs, 1);
      } else {
        W[i] = likelihood(n, X[i], y, 0, Rinv, Nobs, 1);
      }
      normaliser += W[i];
    }
    for (long i = 0; i < N; i++) {
      W[i] /= normaliser;
    }

    ESS = 0;
    for (long i = 0; i < N; i++) {
      ESS += W[i] * W[i];
    }
    ESS = (double) 1.0 / ESS;

    /* Resample */
    resample(N, W, ind, rng);
    for (long i = 0; i < N; i++) {
      X_res[i] = X[ind[i]];
    }

    /* Calculate the output: posterior mean */
    x_hat = 0;
    for (long i = 0; i < N; i++) {
      x_hat += X_res[i] / (double) N;
    }
    x_hats[n] = x_hat;
    /* Calculate the output: posterior variance */
    p_hat = 0;
    for (long i = 0; i < N; i++) {
      diff = X_res[i] - x_hat;
      p_hat += diff * diff / (double) N;
    }

    /* Mutate */
    for (long i = 0; i < N; i++)
      X[i] = X_res[i] + gsl_ran_gaussian(rng, sig_std);

    fprintf(bpf_out, "%i %f %f %f\n", n, x_hat, p_hat, ESS);
    fflush(bpf_out);

    if (floor(n * (double) 50 / data_length) > bar_segment_count) {
      printf("█");
      fflush(stdout);
      bar_segment_count++;
    }
  }
  filtering_time[0] = (double) (clock() - start) / CLOCKS_PER_SEC / (double) data_length;
  filtering_time[1] = 0;
  filtering_time[2] = 0;
  filtering_time[3] = 0;
  printf(" %5.2f sec\n", filtering_time[0]);

  fclose(bpf_out);

  free(X);
  free(X_res);
  free(W);
  free(ind);
  gsl_rng_free(rng);
}

double likelihood(int n, double x, double *y, int band, double *Rinv, int Nobs, double det) {

  double diff;
  double log_likelihood = 0;
  for (int j = 0; j < Nobs; j++) {
    diff = y[n * Nobs + j] - x;
    log_likelihood += (y[n * Nobs + j] - x) * Rinv[j * Nobs + j] * diff;
  }
  log_likelihood /= -(double) 2.0;
  return exp(log_likelihood) / det; // / sqrt(pow((2 * M_PI), Nobs) * det);

}

double full_likelihood(int n, double x, double *y, double *Rinv, int Nobs, double det) {

  double diff;
  double log_likelihood = 0;
  for (int j = 0; j < Nobs; j++) {
    diff = y[n * Nobs + j] - x;
    for (int i = 0; i < Nobs; i++) {
      log_likelihood += (y[n * Nobs + i] - x) * Rinv[j * Nobs + i] * diff;
    }
  }
  log_likelihood /= -(double) 2.0;
  return exp(log_likelihood); // / sqrt(pow((2 * M_PI), Nobs) * det);

}

void random_permuter(long *permutation, long N, gsl_rng *r) {

  for (long i = 0; i < N; i++)
    permutation[i] = i;

  long j;
  long tmp;
  for (long i = N - 1; i > 0; i--) {
    j = gsl_rng_uniform_int(r, i + 1);
    tmp = permutation[j];
    permutation[j] = permutation[i];
    permutation[i] = tmp;
  }

}

void resample(long size, double *w, long *ind, gsl_rng *r) {

  /* Generate the exponentials */
  double *e = (double*) malloc((size + 1) * sizeof(double));
  double g = 0;
  for (long i = 0; i <= size; i++) {
    e[i] = gsl_ran_exponential(r, 1.0);
    g += e[i];
  }
  /* Generate the uniform order statistics */
  double *u = (double *) malloc((size + 1) * sizeof(double));
  u[0] = 0;
  for (long i = 1; i <= size; i++)
    u[i] = u[i - 1] + e[i - 1] / g;

  /* Do the actual sampling with inverse cdf */
  double cdf = w[0];
  long j = 0;
  for (long i = 0; i < size; i++) {
    while (cdf < u[i + 1]) {
      j++;
      cdf += w[j];
    }
    ind[i] = j;
  }

  free(e);
  free(u);
}
