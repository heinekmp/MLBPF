/*
 * BIG_DATA.c
 *
 *  Created on: 3 Sep 2020
 *      Author: heine
 */

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "kalman_filter.h"
#include "particle_filters.h"

#define N_SETTINGS 10
#define N_SCALES 11

void init_with_zeros(double* array, long length);
void scale_sample_sizes(double scaler, long sample_size_collection[N_SETTINGS][2], int n_settings);
void particle_allocation_time_matching(int bpf_sample_size, double std_sig, double obs_std, int Nobs, int data_length, double x0, double *y, double *Rinv1, double *Rinv0, gsl_matrix *R);

int main(void) {
  
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng,clock());
  
  double scaler;
  int N_top = 50;
  int N_bpf = 250;
  int run_time_matching = 0;
  int data_length = 50;
  int Nobs = 500;
  double sig_std = .1;
  double obs_std = 1;
  double x0 = 0;
  double p0 = sig_std * sig_std;
  double sample_size_coeff = (double)68000 / (double) 250;
  
  /* -----------------------------------------------------------------------------------
   *
   * Create data
   *
   -----------------------------------------------------------------------------------*/
  double *x = (double*) malloc(data_length * sizeof(double));
  double *y = (double*) malloc(data_length * Nobs * sizeof(double));
  double *noise = (double*) malloc(Nobs * sizeof(double));
  gsl_matrix * A;
  
  double x_prev = 0;
  
  /* Create the measurement covariance square root */
  A = gsl_matrix_alloc(Nobs, Nobs);
  for (int i = 0; i < Nobs; i++) {
    for (int j = 0; j < Nobs; j++) {
      /* Uncomment for identity matrix */
      /*
       *       if (i == j) {
       *         gsl_matrix_set(A, i, j, 1);
       *       } else {
       *         gsl_matrix_set(A, i, j, 0);
       } */
//      gsl_matrix_set(A, i, j, gsl_ran_gaussian(rng, 1));
      gsl_matrix_set(A, i, j, gsl_rng_uniform(rng));
    }
  }
  
  /* Calculate the observation covariance */
  gsl_matrix * R = gsl_matrix_alloc(Nobs, Nobs);
  double tmp;
  double decay = 2; // Decay of correlations
  FILE * R_out = fopen("R_matrix.txt", "w");
  for (int i = 0; i < Nobs; i++) {
    for (int j = 0; j < Nobs; j++) {
      tmp = 0;
      for (int k = 0; k < Nobs; k++) {
        tmp += gsl_matrix_get(A, i, k) * gsl_matrix_get(A, j, k);
      }
      tmp = obs_std * obs_std * exp(-decay * (double) abs(i - j)) * tmp;
      gsl_matrix_set(R, i, j, tmp);
      fprintf(R_out, "%e ", tmp);
    }
    fprintf(R_out, "\n");
  }
  fclose(R_out);
  
  // Copy R to A
  for (int i = 0; i < Nobs; i++) {
    for (int j = 0; j < Nobs; j++) {
      gsl_matrix_set(A, i, j, gsl_matrix_get(R, i, j));
    }
  }
  gsl_linalg_cholesky_decomp1(A);
  
  printf("Obsevation covariance written in R_matrix.txt\n");
  
  /* Main loop for generating the signal and the data */
  FILE *data_file = fopen("data.txt", "w");
  for (int n = 0; n < data_length; n++) {
    
    x[n] = x_prev + gsl_ran_gaussian(rng, sig_std);
    x_prev = x[n];
    fprintf(data_file, "%i %e", n, x[n]);
    
    /* Uncorrelated observation noise */
    for (int i = 0; i < Nobs; i++) {
      noise[i] = gsl_ran_gaussian(rng, obs_std);
    }
    /* Make the noise correlated by multiplying by A */
    for (int i = 0; i < Nobs; i++) {
      y[n * Nobs + i] = x[n];
      for (int j = 0; j <= i; j++)
        y[n * Nobs + i] += gsl_matrix_get(A, i, j) * noise[j];
      fprintf(data_file, " %e,", y[n * Nobs + i]);
    }
    fprintf(data_file, "\n");
    
  }
  fclose(data_file);
  printf("Signal and data of length %i generated and written in data.txt\n", data_length);
  
  /*
   * Calculate the inverse observation covariances
   */
  printf("Computing the observation covariance inverse...\n");
  fflush(stdout);
  clock_t start = clock();
  
  /* Observation noise covariance */
  gsl_vector * eval = gsl_vector_alloc(Nobs);
  gsl_matrix * evec = gsl_matrix_alloc(Nobs, Nobs);
  double * Rinv1 = (double*) malloc(Nobs * Nobs * sizeof(double));
  gsl_eigen_symmv_workspace * workspace = gsl_eigen_symmv_alloc(Nobs);
  
  /* Compute the eigen decomposition of R to get the inverse */
  gsl_eigen_symmv(R, eval, evec, workspace);
  FILE* rinv_out = fopen("R_inv.txt", "w");
  for (int i = 0; i < Nobs; i++) {
    for (int k = 0; k < Nobs; k++) {
      tmp = 0;
      for (int j = 0; j < Nobs; j++) {
        tmp += gsl_matrix_get(evec, i, j) * gsl_matrix_get(evec, k, j)
        / gsl_vector_get(eval, j);
        
      }
      Rinv1[i * Nobs + k] = tmp;
      fprintf(rinv_out, "%e, ", tmp);
    }
    fprintf(rinv_out, "\n");
  }
  fclose(rinv_out);
  printf("Observation covariance inverse computed and written in R_inv.txt\n");
  
  gsl_matrix_free(evec);
  gsl_vector_free(eval);
  gsl_eigen_symmv_free(workspace);
  
  /* Limit the observation correlation to a band and ... */
  int band = 0; // Note we only consider diagonal, i.e. band = 0.
  gsl_matrix * R_low = gsl_matrix_alloc(Nobs, Nobs);
  for (int i = 0; i < Nobs; i++) {
    for (int j = 0; j < Nobs; j++) {
      tmp = abs(i - j) <= band ? gsl_matrix_get(R, i, j) : 0;
      gsl_matrix_set(R_low, i, j, tmp);
    }
  }
  /* ...take inverse */
  gsl_vector * eval_low = gsl_vector_alloc(Nobs);
  gsl_matrix * evec_low = gsl_matrix_alloc(Nobs, Nobs);
  gsl_eigen_symmv_workspace * workspace_low = gsl_eigen_symmv_alloc(Nobs);
  double * Rinv0 = (double*) malloc(Nobs * Nobs * sizeof(double));
  
  gsl_eigen_symmv(R_low, eval_low, evec_low, workspace_low);
  
  FILE * band_R_out = fopen("R_band_inv.txt", "w");
  for (int i = 0; i < Nobs; i++) {
    for (int k = 0; k < Nobs; k++) {
      tmp = 0;
      for (int j = 0; j < Nobs; j++) {
        tmp += gsl_matrix_get(evec_low, i, j) * gsl_matrix_get(evec_low, k, j)
        / gsl_vector_get(eval_low, j);
        
      }
      Rinv0[i * Nobs + k] = tmp;
      fprintf(band_R_out, "%e ", tmp);
    }
    fprintf(band_R_out, "\n");
  }
  fclose(band_R_out);
  printf(
         "Inverse of the diagonal covariance approximation computed and written in R_band_inv.txt\n");
  
  gsl_matrix_free(evec_low);
  gsl_vector_free(eval_low);
  gsl_eigen_symmv_free(workspace_low);
  
  printf("elapsed %5.2f sec\n", (double) (clock() - start) / CLOCKS_PER_SEC);
  
  if (run_time_matching == 1) {
    /*
     * Particle allocation time matching
     *
     * This part of the code should be run when trying to find the correct N0 sample sizes to match
     * the BPF running time
     */
    
    particle_allocation_time_matching(N_bpf, sig_std, obs_std, Nobs, data_length, x0, y, Rinv1, Rinv0, R);
    
    printf("Particle allocation time matching done! Exiting...\n");
    return 0; // the code exits
  }
  
  /* ------------------------------------------------------------
   * Kalman filter
   * --------------------------------------------------------- */
  double *x_hat = (double*) malloc(data_length * sizeof(double));
  double *p_hat = (double*) malloc(data_length * sizeof(double));
  
  kalman_filter(x, y, Rinv1, sig_std, obs_std, x_hat, p_hat, data_length, x0, p0, Nobs);
  
  /* Write KF results in a file */
  FILE *kf_out = fopen("kf_out.txt", "w");
  for (int i = 0; i < data_length; i++)
    fprintf(kf_out, "%i %f %f %f\n", i, x[i], x_hat[i], p_hat[i]);
  fclose(kf_out);
  
  /*
   * Top level loop
   * */
  double scalers[N_SCALES] = {0.5,1,2,3,4,5,6,7,8,9,10};
  short N_levels = 2;
  long sample_sizes[2] = {0,0};
  long sample_size[1] = { N_bpf }; // This is for one level BPF
  
  double *x_hat_bpf = (double*) malloc(data_length * sizeof(double));
  double *errors = (double *) malloc((N_SETTINGS + 1) * sizeof(double));
  double *times = (double *) malloc((N_SETTINGS + 1) * sizeof(double));
  init_with_zeros(errors, N_SETTINGS + 1);
  init_with_zeros(times, N_SETTINGS + 1);
  
  double error;
  FILE * error_out = fopen("error.txt", "w");
  
  double filtering_time[4];
  double worst_case_sign_ratio[1] = { 100 }; // This is basically sustitute for "Inf"
  long unilevel_sample_size = N_bpf;
  long error_matched_unilevel_sample_size = 7 * N_bpf;
  int N1, N0;
  double step;
  
  /*
   * MAIN LOOP FOR TOP LEVEL MONTE CARLO
   */
  for (int i = 0; i < N_top; i++) {
    
    printf("Top level iteration %i\n", i);
    
    for (int k = 0; k < N_SCALES; k++) {
      
      scaler = scalers[k];
      
      printf("Scaler %f\n", scaler);
      
      
      /* Run the basic BPF (time matched) */
      sample_size[0] = (long) round(unilevel_sample_size * scaler);
      bootstrapfilter(y, sig_std, obs_std, data_length, x0, Nobs, sample_size[0],
                      x_hat_bpf, filtering_time, Rinv1, 1);
      
      /* Sum of squared errors over the whole signal */
      error = 0;
      for (int n = 0; n < data_length; n++) {
        error += (x_hat[n] - x_hat_bpf[n]) * (x_hat[n] - x_hat_bpf[n]);
      }
      
      fprintf(error_out, "%e %i %lu %f %i %e %e %f %e %f\n",
              error / (double) data_length, 0, sample_size[0], filtering_time[0], 0,
              filtering_time[1], filtering_time[2], worst_case_sign_ratio[0],
              filtering_time[3], scaler);
      fflush(error_out);
      
      if(scaler < 8) {
        /* Run the basic BPF (error matched) */
        sample_size[0] = (long) round(error_matched_unilevel_sample_size * scaler);
        bootstrapfilter(y, sig_std, obs_std, data_length, x0, Nobs, sample_size[0],
                        x_hat_bpf, filtering_time, Rinv1, 1);
        
        /* Sum of squared errors over the whole signal */
        error = 0;
        for (int n = 0; n < data_length; n++) {
          error += (x_hat[n] - x_hat_bpf[n]) * (x_hat[n] - x_hat_bpf[n]);
        }
        
        fprintf(error_out, "%e %i %lu %f %i %e %e %f %e %f\n",
                error / (double) data_length, 0, sample_size[0], filtering_time[0], -1,
                filtering_time[1], filtering_time[2], worst_case_sign_ratio[0],
                filtering_time[3], scaler);
        fflush(error_out);
      }
      
      /* Iterate over different configurations */
      N1 = 0;
      for (int j = 0; j < N_SETTINGS; j++) {
        
        step = (double)(N_bpf - 5) / (double)(N_SETTINGS-1);
        N1 = (int) j * step;
        N0 = (int) floor((N_bpf-N1)*sample_size_coeff);
        worst_case_sign_ratio[0] = 100;
        
        sample_sizes[0] = N0 * scaler;
        sample_sizes[1] = N1 * scaler;
        
        printf("N0 = %lu, N1 = %lu, N = %lu\n", sample_sizes[0], sample_sizes[1], sample_sizes[0]+sample_sizes[1]);
        
        MLbootstrapfilter(y, sig_std, obs_std, data_length, x0, Nobs, N_levels,
                          sample_sizes, x_hat_bpf, filtering_time, Rinv0, Rinv1,
                          worst_case_sign_ratio);
        
        /* Sum of squared errors over the whole signal */
        error = 0;
        for (int n = 0; n < data_length; n++) {
          error += (x_hat[n] - x_hat_bpf[n]) * (x_hat[n] - x_hat_bpf[n]);
        }
        
        fprintf(error_out, "%e %lu %lu %f %i %e %e %f %e %f\n",
                error / (double) data_length, sample_sizes[0],
                sample_sizes[1], filtering_time[0], j+1, filtering_time[1],
                filtering_time[2], worst_case_sign_ratio[0], filtering_time[3], scaler);
        
      }
      fflush(error_out);
    }
    
  }
  
  fclose(error_out);
  
  printf("Results of all BPF runs written in error.txt\n");
  printf("Error summary written in error_summary.txt\n");
  
  return 0;
}

void init_with_zeros(double *array, long length) {
  for (int i = 0; i < length; i++)
    array[i] = 0;
}

void scale_sample_sizes(double scaler, long sample_size_collection[N_SETTINGS][2], int n_settings) {
  
  for (int i = 0; i < n_settings; i++)
    for (int j = 0; j < 1; j++)
      sample_size_collection[i][j] = (long) round(scaler * sample_size_collection[i][j]);
  
}

void particle_allocation_time_matching(int bpf_sample_size, double std_sig, double obs_std, int Nobs, int data_length, double x0, double *y, double *Rinv1, double *Rinv0, gsl_matrix *R) {
  
  int N1_step = 20;
  int N0_step = 10000;
  int N0_init = 10;
  int N_top = 5;
  double time_per_iteration;
  double time_tolerance = 1.00; //
  double filtering_time[1];
  double worst_case_sign_ratio[1] = { 100 };
  
  double *x_hat_bpf = (double*) malloc(data_length * sizeof(double));
  
  /* Find the reference time per iteration for BPF */
  double bpf_time_per_iteration = 0;
  
  for(int nmc = 0; nmc < N_top; nmc++) {
    bootstrapfilter(y, std_sig, obs_std, data_length, x0, Nobs, bpf_sample_size,
                    x_hat_bpf, filtering_time, Rinv1, 1);
    bpf_time_per_iteration += filtering_time[0] / (double) N_top;
  }
  
  printf("Time per iteration (BPF): %0.5f\n",bpf_time_per_iteration);
  fflush(stdout);
  
  long sample_sizes[2] = {0,0};
  int N0;
  short direction;
  
  FILE* out = fopen("time_matched_sample_sizes.txt","w");
  
  /* For given L0 mesh, iterate over N1 */
  for(int N1 = 0; N1 <= bpf_sample_size ; N1 = N1 + N1_step) {
    
    N0 = N0_init;
    N0_step = 10000;
    
    /* For given L0 mesh and N1 iterate over increasing N0 until MLBPF time per iteration exceeds
     BPF time per iteration */
    time_per_iteration = 0;
    
    direction = 1; // up
    while(1) {
      
      sample_sizes[0] = N0;
      sample_sizes[1] = N1;
      
      /* Estimate the time per iteration by averaging N_top runs */
      time_per_iteration = 0;
      for(int nmc = 0; nmc < N_top; nmc++) {
        
        MLbootstrapfilter(y, std_sig, obs_std, data_length, x0, Nobs, 2,
                          sample_sizes, x_hat_bpf, filtering_time, Rinv0, Rinv1,
                          worst_case_sign_ratio);
        
        time_per_iteration += filtering_time[0] / (double) N_top;
      }
      
      if (direction == 1 && time_per_iteration > bpf_time_per_iteration * time_tolerance && N0_step > 10) {
        direction = -1; // start going down
        N0_step /= 2; // halve the step
        
      } else if (direction == -1 && time_per_iteration < bpf_time_per_iteration * 1.00 && N0_step > 10) {
        direction = 1;
        N0_step /= 2;
      } else if (N0_step < 10) {
        break;
      }
      N0 = N0 + direction * N0_step;
      
      if(N0 < 1) { // Make sure N0 remains positive and go up if this happens
        N0 = 1;
        direction = 1;
      }
      
    }
    fprintf(out,"%i %i %i %i\n",N0-N0_step,N1,0,0);
    fflush(out);
  }
  
  fclose(out);
  free(x_hat_bpf);
}
