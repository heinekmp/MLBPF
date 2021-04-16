//
//  main.c
//  ODE_MODEL
//
//  Created by Kari Heine on 28/10/2020.
//  Copyright © 2020 Kari Heine. All rights reserved.
//

#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

#include "particle_filters.h"

#define MAX_TEST_MESH_SIZE 4000 // Mesh size for BPF and MLBPF level 1 (the truth mesh size)
#define N_SETTINGS 500 // Number of different scenarios
#define N_SCALES 1 // Scaling samplesizes to study convergence
#define N_MEASUREMENTS 2 // Number of places where the beam deflection is measured

void generate_signal(double* x, int len, double std_sig, double x0, gsl_rng* rng, double L);
void create_dense_solver_matrix(double* A, int n);
void init_with_zeros(double* array, long length);
void make_lapack_band_matrix(double *A, int n, double *ab, int ku, int kl);
double compute_estimation_error(double* x_hat, double* x_ref, int len);
void particle_allocation_time_matching(int max_mesh_size, int* level_0_mesh_sizes, int n_level_0_mesh_sizes, int bpf_sample_size, double std_sig, double obs_std, int data_length, double x0, double *y, int kl, int ku, double* ab_bpf, int ldab, double w, double d, double E, double I, double h_bpf, double *ab0, double h1, double L, double* meass, int N_meas);
void get_settings(int** settings);
void write_beam_in_file(int data_length, int true_mesh_size, double h, double* beam);

int main(void) {
  
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(rng,clock());
//  gsl_rng_set(rng,42);
  
  /* Global parameters */
  /* ----------------- */
  int N_top = 250; // Top leve Monte Carlo iterations
  int kl = 3; // Lower diagonal bandwidth for the ODE solver
  int ku = 3; // upper diagonal bandwidth for the ODE solver
  int ldab = 2 * kl + ku + 1; // parameter for Lapack DGBSV
  int run_time_matching = 0; // This toggle (0 or 1) determines if we want to run time matching or filtering
  int N_ref = 100000; // Sample size for the reference filter
  
  /* Signal and observations */
  /* ----------------------- */
  int data_length = 50; // number of time steps
  double x0 = 2; // initial state
  double std_sig = 0.02; // signal noise standard deviation
  double obs_std = 0.0002; // observation noise standard deviation
  
  // Data arrays for the signal and observations
  double* x = (double*) malloc(data_length * sizeof(double));
  double* y = (double*) malloc(N_MEASUREMENTS * data_length * sizeof(double));
  FILE * data_out;
  
  /* Euler-Bernoulli beam parameters */
  /* ------------------------------- */
  double L = 4; // Length (m)
  double w = 0.3; // Width (m)
  double d = 0.03; // thickness (m)
  double E = 1e10; // Young's modulus
  double I = w * d * d * d / (double) 12; // Inertia
  double measurements[N_MEASUREMENTS] = {1, 1.75}; // deflection measurement points along the beam (m)
  int minds[N_MEASUREMENTS]; // mesh point indices for the deflection measurement points
  
  /* Beam solver setup for data generation */
  /* ------------------------------------- */
  int true_mesh_size = MAX_TEST_MESH_SIZE;
  double* A = (double*) malloc(MAX_TEST_MESH_SIZE * MAX_TEST_MESH_SIZE * sizeof(double));
  int* ipiv = (int*) malloc(MAX_TEST_MESH_SIZE * sizeof(int));
  double* ab = (double*) malloc(ldab * MAX_TEST_MESH_SIZE * sizeof(double));
  double* ab_copy = (double*) malloc(ldab * MAX_TEST_MESH_SIZE * sizeof(double));
  double* signal_payload = (double*) malloc(MAX_TEST_MESH_SIZE * data_length * sizeof(double));
  double h = L / (double)(true_mesh_size-1);
  int ldb = true_mesh_size, info = 0;
  
  // Calculate the indices for the deflection measurement points.
  for(int i = 0; i < N_MEASUREMENTS; i++) {
    minds[i] = (int)((measurements[i] / L) * (double) true_mesh_size);
  }
  
  create_dense_solver_matrix(A, true_mesh_size);
  
  /* Convert the design matrix to general band representation */
  init_with_zeros(ab, ldab * true_mesh_size);
  make_lapack_band_matrix(A, true_mesh_size, ab, ku, kl);
  
  /* COMPARISON SETUP */
  /* ---------------- */
  FILE * error_out = fopen("error.txt", "w");
  double bpf_MSE = 0;
  double mlbpf_MSE = 0;
  double scaler = 1;
  double bpf_MSE_error_matched = 0;
  
  /* BPF SETUP */
  /* --------- */
  double *x_hat_bpf = (double*) malloc(data_length * sizeof(double));
  double *x_ref = (double*) malloc(data_length * sizeof(double));
  double *p_ref = (double*) malloc(data_length * sizeof(double));
  FILE* ref_out;
  double filtering_time[4]; // array for time recording
  int N = 500;
  int mesh_size_bpf = MAX_TEST_MESH_SIZE;
  double* ab_bpf = ab;
  int* ipiv_bpf = ipiv;
  double h_bpf = L / (double) (mesh_size_bpf-1);
  
  create_dense_solver_matrix(A, mesh_size_bpf);
  init_with_zeros(ab_bpf, ldab*mesh_size_bpf);
  make_lapack_band_matrix(A, mesh_size_bpf, ab_bpf, ku, kl);
  
  /* MLBPF SETUP */
  /* ----------- */
  double worst_case_sign_ratio[1] = { 100 }; // This is basically sustitute for "Inf"
  double *x_hats_mlbpf = (double*) malloc(data_length * sizeof(double));
  int mesh_sizes[2] = {10, 500}; // these values will be overwritten
  int N0, N1;
  
  double* ab0 = (double*) malloc(ldab * MAX_TEST_MESH_SIZE * sizeof(double));
  double* ab1 = ab;
  int* ipiv0 = ipiv;
  int* ipiv1 = ipiv;
  double h0;
  
  init_with_zeros(ab0, ldab * MAX_TEST_MESH_SIZE);
  init_with_zeros(ab1, ldab * MAX_TEST_MESH_SIZE);
  
  // We can create the solver matrices for level 1 only which has fixed mesh size
  create_dense_solver_matrix(A, MAX_TEST_MESH_SIZE);
  make_lapack_band_matrix(A, MAX_TEST_MESH_SIZE, ab1, ku, kl);
  double h1 = L / (double)(MAX_TEST_MESH_SIZE - 1);
  
  if(run_time_matching) {
    /*
     * Particle allocation time matching
     *
     * This part of the code should be run when trying to find the correct N0 sample sizes to match the
     * BPF running time
     */
    int max_mesh_size = MAX_TEST_MESH_SIZE;
    int level_0_mesh_sizes[20] = {100, 105, 110, 115, 120, 130, 140, 150, 160, 180, 200, 220, 240, 260, 280, 320, 360, 400, 440, 500};
    int n_level_0_mesh_sizes = 20;
    
    generate_signal(x, data_length, std_sig, x0, rng, L);
    create_payload(data_length, true_mesh_size, signal_payload, w, d, x, h, E, I, L);
    
    array_copy(ab, ab_copy, ldab * true_mesh_size); // Restore undamaged band matrix
    
    // Solve the Euler-Bernoulli beam
    dgbsv_(&true_mesh_size, &kl, &ku, &data_length, ab_copy, &ldab, ipiv, signal_payload, &ldb, &info);
        
    // Generate measurements
    for(int i = 0; i < data_length; i++)
      for(int j = 0; j < N_MEASUREMENTS; j++)
        y[i + j * data_length] = signal_payload[i * true_mesh_size + minds[j]] + gsl_ran_gaussian(rng, obs_std);
    
    particle_allocation_time_matching(max_mesh_size, level_0_mesh_sizes, n_level_0_mesh_sizes, N, std_sig, obs_std, data_length, x0, y, kl, ku, ab_bpf, ldab, w, d, E, I, h_bpf, ab0, h1, L, measurements, N_MEASUREMENTS);
    
    printf("Particle allocation time matching done! Exiting...\n");
    return 0; // the code exits
  }
  
  /*
   *  THIS IS THE ACTUAL FILTER COMPARISON PART
   */
  
  // Settings are hard coded here. TODO: This should be changed to reading a file.
  int** settings = (int **)malloc(N_SETTINGS * sizeof(int*));
  for(int i = 0; i < N_SETTINGS; i++)
    settings[i] = (int *) malloc(4*sizeof(int));
  int sample_sizes[2] = {0,0};
  get_settings(settings);
  
  // Top level iteration
  for(int nmc = 0; nmc < N_top; nmc++) {
    
    printf("Top level iteration %i out of %i.*****************************\n", nmc + 1, N_top);
    
    /* To save computational time and also to take into account the possibility that a particular
     data set is more favourable to one or the other algorithm, we create new data every nth top
     level iteration, and compute the reference filter.
     */
    if (nmc % 10 == 0) {
      
      // Create the signal
      generate_signal(x, data_length, std_sig, x0, rng, L);

      printf("Signal generated\n");
      fflush(stdout);

      create_payload(data_length, true_mesh_size, signal_payload, w, d, x, h, E, I, L);

      printf("Payload created\n");
      fflush(stdout);
      
      array_copy(ab, ab_copy, ldab * true_mesh_size);
      
      // Solve the Euler-Bernoulli beam
      dgbsv_(&true_mesh_size, &kl, &ku, &data_length, ab_copy, &ldab, ipiv, signal_payload, &ldb, &info);
      
      printf("\nBeam solved. Info: %i\n\n",info);
      fflush(stdout);
      
      // write_beam_in_file(data_length, true_mesh_size, h, signal_payload);
      
      // Generate the noisy measurements
      data_out = fopen("data.txt","w");
      for(int i = 0; i < data_length; i++) {
        
        // Noisy measurements for one time step but all measurement points
        for(int j = 0; j < N_MEASUREMENTS; j++) {
          y[i + j*data_length] = signal_payload[i * true_mesh_size + minds[j]] + gsl_ran_gaussian(rng, obs_std);
        }
        
        fprintf(data_out,"%i %e ",i, x[i]); // write true signal into a file
        for(int j = 0; j < N_MEASUREMENTS; j++) {
          // write noisy and noiseless observation into a file
          fprintf(data_out,"%6.6e %6.6e ", y[i+j*data_length], signal_payload[i * true_mesh_size + minds[j]]);
        }
        fprintf(data_out,"\n");
      }
      fclose(data_out);
      
      printf("Data generated\n");
      
      /* Compute a reference solution */
      printf("Reference filter with N_ref = %i\n", N_ref);
      fflush(stdout);
      bootstrapfilter(y, std_sig, obs_std, data_length, x0, N_ref, x_ref, p_ref, filtering_time, mesh_size_bpf, kl, ku, ab_bpf, ldab, ipiv_bpf, mesh_size_bpf, info, w, d, E, I, h_bpf, L, measurements, (int)N_MEASUREMENTS);
      
      // Write the reference filter output into a file with columns:
      // step index, posterior mean, posterior variance, true signal, observation
      ref_out = fopen("ref_filter_out.txt","w");
      for(int i = 0; i < data_length;i++)
        fprintf(ref_out,"%i %e %e %e %e\n",i,x_ref[i],p_ref[i],x[i],y[i]);
      fclose(ref_out);
      
      // return 0; // option to exit after reference filter may be useful in debugging
    
    }
    
    /* Run the benchmark BPF */
    // NB: This is not the reference, but a comparison filter which is time is matched to MLBPF
    bootstrapfilter(y, std_sig, obs_std, data_length, x0, N, x_hat_bpf, NULL, filtering_time, mesh_size_bpf, kl, ku, ab_bpf, ldab, ipiv_bpf, mesh_size_bpf, info, w, d, E, I, h_bpf, L,  measurements, (int)N_MEASUREMENTS);
    
    // bpf_MSE = compute_estimation_error(x_hat_bpf,x,data_length); // compute the error w.r.t. the truth
    bpf_MSE = compute_estimation_error(x_hat_bpf,x_ref,data_length);
    
    // Write the results in a file
    fprintf(error_out, "%e %i %i %f %i %e %e %f %e %f %i\n", bpf_MSE, 0, N, filtering_time[0], 0,         filtering_time[1], filtering_time[2], worst_case_sign_ratio[0], filtering_time[3], scaler, mesh_size_bpf);
    
    /* Compute the error matched BPF */
    // The ERROR matched filter, i.e. it produces roughly the same level of error, but hopefully takes
    // much longer to achieve it.
    printf("Error matched filter\n");
    bootstrapfilter(y, std_sig, obs_std, data_length, x0, 4*N, x_hat_bpf, NULL, filtering_time, mesh_size_bpf, kl, ku, ab_bpf, ldab, ipiv_bpf, mesh_size_bpf, info, w, d, E, I, h_bpf, L,  measurements, (int)N_MEASUREMENTS);
    
    //  bpf_MSE = compute_estimation_error(x_hat_bpf,x,data_length); // compute the error w.r.t. the truth
    bpf_MSE_error_matched = compute_estimation_error(x_hat_bpf,x_ref,data_length);
    
    // Write the results in a file
    fprintf(error_out, "%e %i %i %f %i %e %e %f %e %f %i\n", bpf_MSE_error_matched, 0, N, filtering_time[0], -1,         filtering_time[1], filtering_time[2], worst_case_sign_ratio[0], filtering_time[3], scaler, mesh_size_bpf);
    
    /* Iterate over different mesh size and particle allocation configurations */
    for(int i_setting = 0; i_setting < N_SETTINGS; i_setting++) {
      
      // Rename
      sample_sizes[0] = settings[i_setting][0];
      sample_sizes[1] = settings[i_setting][1];
      mesh_sizes[0] = settings[i_setting][2];
      mesh_sizes[1] = settings[i_setting][3];
      
      // Ensure odd number of particles
      sample_sizes[0] = (sample_sizes[0] + sample_sizes[1]) % 2 == 0 ? sample_sizes[0] + 1 : sample_sizes[0];
      
      init_with_zeros(ab0, ldab * mesh_sizes[0]);
      create_dense_solver_matrix(A, mesh_sizes[0]);
      make_lapack_band_matrix(A, mesh_sizes[0], ab0, ku, kl);
      h0 = L / (double)(mesh_sizes[0] - 1);
      
      printf("Setting %i out of %i\n",i_setting+1,N_SETTINGS);
      fflush(stdout);
      
      MLbootstrapfilter(y, std_sig, obs_std, data_length, x0, sample_sizes, worst_case_sign_ratio, x_hats_mlbpf, filtering_time, mesh_sizes, kl, ku, ab0, ab1, ldab, ipiv0, ipiv1, mesh_sizes[0], mesh_sizes[1], info, w, d, E, I, h0, h1, L, measurements, (int)N_MEASUREMENTS);
    
      //  mlbpf_MSE = compute_estimation_error(x_hats_mlbpf,x,data_length);
      mlbpf_MSE = compute_estimation_error(x_hats_mlbpf,x_ref,data_length);
      
      N0 = sample_sizes[0];
      N1 = sample_sizes[1];
      
      fprintf(error_out, "%e %i %i %f %i %e %e %f %e %f %i\n", mlbpf_MSE, N0, N1, filtering_time[0], i_setting + 1,         filtering_time[1], filtering_time[2], worst_case_sign_ratio[0], filtering_time[3], scaler, mesh_sizes[0]);
      fflush(error_out);
      
    }
  }
  
  /* Free all that was allocated */
  free(x);
  free(y);
  free(A);
  free(ipiv);
  free(ab);
  free(ab_copy);
  free(signal_payload);
  free(x_hat_bpf);
  free(x_ref);
  free(p_ref);
  free(x_hats_mlbpf);
  free(ab0);
  for(int i = 0; i < N_SETTINGS; i++)
    free(settings[i]);
  free(settings);
  
  return 0;
}

void write_beam_in_file(int data_length, int true_mesh_size, double h, double* beam) {
  
  FILE * payload_out = fopen("payload.txt","w");
  for(int i = 0; i < data_length; i++)
    for(int j = 0; j < true_mesh_size; j++)
      fprintf(payload_out, "%5.5e %5.10e \n", j * h, beam[i * true_mesh_size + j]);
  fclose(payload_out);
  printf("Payload written in file\n");
  fflush(stdout);
  
}

void particle_allocation_time_matching(int max_mesh_size, int* level_0_mesh_sizes, int n_level_0_mesh_sizes, int bpf_sample_size, double std_sig, double obs_std, int data_length, double x0, double *y, int kl, int ku, double* ab_bpf, int ldab, double w, double d, double E, double I, double h_bpf, double *ab0, double h1, double L, double* meass, int N_meas) {
  
  int N1_step = 20;
  int N0_step = 10000;
  int N0_init = 10;
  int N_top = 5;
  double time_per_iteration;
  double time_tolerance = 1.00; //
  double filtering_time[4];
  int info = 0;
  int* ipiv = (int *) malloc(max_mesh_size * sizeof(int));
  double worst_case_sign_ratio[1] = { 100 };
  double* ab_copy = (double*) malloc(ldab * max_mesh_size * sizeof(double));
  double* ab0_copy = (double*) malloc(ldab * max_mesh_size * sizeof(double));
  double h0;
  double* A0 = (double*) malloc(MAX_TEST_MESH_SIZE * MAX_TEST_MESH_SIZE * sizeof(double));
  
  /* Find the reference time per iteration for BPF */
  double bpf_time_per_iteration = 0;
  for(int nmc = 0; nmc < N_top; nmc++) {
    
    for(int i = 0; i < ldab * max_mesh_size; i++)
      ab_copy[i] = ab_bpf[i];
    
    bootstrapfilter(y, std_sig, obs_std, data_length, x0, bpf_sample_size, NULL, NULL, filtering_time, max_mesh_size, kl, ku, ab_bpf, ldab, ipiv, max_mesh_size, info, w, d, E, I, h_bpf , L, meass, N_MEASUREMENTS);
    bpf_time_per_iteration += filtering_time[0] / (double) N_top;
  }
  
  printf("Time per iteration (BPF): %0.5f\n",bpf_time_per_iteration);
  fflush(stdout);
  
  int level_0_mesh_size;
  int sample_sizes[2] = {0,0};
  int mesh_sizes[2];
  int N0;
  short direction;
  
  FILE* out = fopen("time_matched_sample_sizes.txt","w");
  
  /* Iterate over level 0 mesh sizes */
  for(int j = 0; j < n_level_0_mesh_sizes; j++) {
    
    level_0_mesh_size = level_0_mesh_sizes[j];
    mesh_sizes[0] = level_0_mesh_size;
    mesh_sizes[1] = max_mesh_size;
    
    // We have to create A0 again for each mesh size
    for(int i = 0; i < ldab * mesh_sizes[0]; i++)
      ab0[i] = 0;
    create_dense_solver_matrix(A0, mesh_sizes[0]);
    make_lapack_band_matrix(A0, mesh_sizes[0], ab0, ku, kl);
    h0 = L / (double)(mesh_sizes[0] - 1);
    
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
          
          for(int i = 0; i < ldab * max_mesh_size; i++)
            ab_copy[i] = ab_bpf[i];
          
          for(int i = 0; i < ldab * level_0_mesh_size; i++)
            ab0_copy[i] = ab0[i];
          
          MLbootstrapfilter(y, std_sig, obs_std, data_length, x0, sample_sizes, worst_case_sign_ratio, NULL, filtering_time, mesh_sizes, kl, ku, ab0_copy, ab_copy, ldab, ipiv, ipiv, mesh_sizes[0], max_mesh_size, info, w, d, E, I, h0, h1, L, meass, N_MEASUREMENTS);
          //          printf("TIMES: %e %e %e %e\n", filtering_time[0], filtering_time[1], filtering_time[2], filtering_time[3]);
          //          fflush(stdout);
          
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
      fprintf(out,"%i %i %i %i\n",N0-N0_step,N1,mesh_sizes[0],mesh_sizes[1]);
      fflush(out);
    }
  }
  fclose(out);
  
  free(ipiv);
  free(ab_copy);
  free(ab0_copy);
  free(A0);
}

double compute_estimation_error(double* x_hat, double* x_ref, int len) {
  
  double MSE = 0;
  
  for(int i = 0; i < len; i++) {
    MSE += (x_hat[i]-x_ref[i])*(x_hat[i]-x_ref[i]) / (double) len;
  }
  return MSE;
}

void init_with_zeros(double *array, long length) {
  for (int i = 0; i < length; i++)
    array[i] = 0;
}

void generate_signal(double* x, int len, double std_sig, double x0, gsl_rng* rng, double L) {
  
  double lower_bound = 0.25;
  double upper_bound = L - 0.25;
  x[0] = x0;
  
  for (int i = 1; i < len; i++){
    x[i] = x[i - 1] + gsl_ran_gaussian(rng, std_sig);
    //    x[i] = 0.5 / ((double)1.0 + exp(-(double)10.0 * (x[i - 1] - 1))) + (double) 0.75 + gsl_ran_gaussian(rng, std_sig);
    if(x[i] < lower_bound)
      x[i] = lower_bound + lower_bound-x[i];
    if(x[i] > upper_bound)
      x[i] = upper_bound - (x[i]-upper_bound);
  }
  
}

void make_lapack_band_matrix(double *A, int n, double *ab, int ku, int kl) {
  
  int lower, upper, baserow,baserowcand ;
  int n_rows = 2 * kl + ku + 1;
  
  for (int j = 0; j < n; j++) { // iterate columns
    
    lower = j - ku > 0 ? j - ku : 0;
    upper = j + kl < n ? j + kl : n-1;
    
    baserowcand = n_rows - upper - 1;
    baserow = baserowcand < n_rows - (ku+kl+1)? n_rows - (ku+kl+1) : baserowcand;
    
    for (int i = lower; i <= upper; i++) {
      ab[n_rows * j + baserow + i - lower] = A[j * n + i];
    }
  }
  
}

void create_dense_solver_matrix(double* A, int n) {
  
  double pattern[5] = { 1, -4, 6, -4, 1 };
  /* Make sure there are zeros */
  for(int i = 0; i < n*n; i++)
    A[i] = 0;
  
  /* First two lines */
  for (int i = 0; i < n; i++) {
    A[i * n] = 0;
    A[i * n + 1] = 0;
  }
  
  A[0] = 16;
  A[1] = -4;
  A[n + 0] = -9;
  A[n + 1] = 6;
  A[2 * n + 0] = (double) 8 / (double) 3;
  A[2 * n + 1] = -4;
  A[3 * n + 0] = -(double) 1 / (double) 4;
  A[3 * n + 1] = (double) 1;
  
  /* Last two lines */
  for (int i = 0; i < n; i++) {
    A[i * n + n - 2] = 0;
    A[i * n + n - 1] = 0;
  }
  
  //  A[(n - 4) * n + n - 2] = (double) 16 / (double) 17;
  A[(n - 4) * n + n - 2] = (double) 1;
  //  A[(n - 4) * n + n - 1] = -(double) 12 / (double) 17;
  A[(n - 4) * n + n - 1] = -(double) 1 / (double) 4;
  //  A[(n - 3) * n + n - 2] = -(double) 60 / (double) 17;
  A[(n - 3) * n + n - 2] = -(double) 4;
  //  A[(n - 3) * n + n - 1] = (double) 96 / (double) 17;
  A[(n - 3) * n + n - 1] = (double) 8 / (double) 3;
  //  A[(n - 2) * n + n - 2] = (double) 72 / (double) 17;
  A[(n - 2) * n + n - 2] = (double) 6;
  //  A[(n - 2) * n + n - 1] = -(double) 156 / (double) 17;
  A[(n - 2) * n + n - 1] = -(double) 9;
  //  A[(n - 1) * n + n - 2] = -(double) 28 / (double) 17;
  A[(n - 1) * n + n - 2] = -(double) 4;
  //  A[(n - 1) * n + n - 1] = (double) 72 / (double) 17;
  A[(n - 1) * n + n - 1] = (double) 16;
  
  /* The rest */
  for (int i = 2; i < n - 2; i++) { // row
    for (int j = 0; j < n; j++) {
      A[j * n + i] = 0;
    }
    for (int j = 0; j < 5; j++)
      A[(j + i - 2) * n + i] = pattern[j];
  }
}

void get_settings(int** settings) {
  
//  /* THIS IS FOR N = 1000 */
//  int hc_settings[N_SETTINGS][4] = {
//  {38000, 0, 100, 4000},
//{36835, 40, 100, 4000},
//{36054, 80, 100, 4000},
//{36439, 120, 100, 4000},
//{35867, 160, 100, 4000},
//{35658, 200, 100, 4000},
//{34954, 240, 100, 4000},
//{34888, 280, 100, 4000},
//{34570, 320, 100, 4000},
//{34237, 360, 100, 4000},
//{33456, 400, 100, 4000},
//{32811, 440, 100, 4000},
//{32127, 480, 100, 4000},
//{30799, 520, 100, 4000},
//{29146, 560, 100, 4000},
//{27180, 600, 100, 4000},
//{24810, 640, 100, 4000},
//{22713, 680, 100, 4000},
//{20024, 720, 100, 4000},
//{17349, 760, 100, 4000},
//{14049, 800, 100, 4000},
//{10800, 840, 100, 4000},
//{7812, 880, 100, 4000},
//{4960, 920, 100, 4000},
//{2408, 960, 100, 4000},
//{36965, 0, 105, 4000},
//{36354, 40, 105, 4000},
//{35552, 80, 105, 4000},
//{34341, 120, 105, 4000},
//{34073, 160, 105, 4000},
//{33840, 200, 105, 4000},
//{33970, 240, 105, 4000},
//{33444, 280, 105, 4000},
//{33079, 320, 105, 4000},
//{32883, 360, 105, 4000},
//{32615, 400, 105, 4000},
//{32173, 440, 105, 4000},
//{31021, 480, 105, 4000},
//{29393, 520, 105, 4000},
//{27453, 560, 105, 4000},
//{25473, 600, 105, 4000},
//{23605, 640, 105, 4000},
//{21639, 680, 105, 4000},
//{19367, 720, 105, 4000},
//{16693, 760, 105, 4000},
//{13229, 800, 105, 4000},
//{10143, 840, 105, 4000},
//{7134, 880, 105, 4000},
//{4543, 920, 105, 4000},
//{2108, 960, 105, 4000},
//{35266, 0, 110, 4000},
//{35461, 40, 110, 4000},
//{35409, 80, 110, 4000},
//{35058, 120, 110, 4000},
//{34531, 160, 110, 4000},
//{33971, 200, 110, 4000},
//{33437, 240, 110, 4000},
//{33026, 280, 110, 4000},
//{32804, 320, 110, 4000},
//{32765, 360, 110, 4000},
//{32550, 400, 110, 4000},
//{31659, 440, 110, 4000},
//{30162, 480, 110, 4000},
//{28254, 520, 110, 4000},
//{26203, 560, 110, 4000},
//{24055, 600, 110, 4000},
//{21965, 640, 110, 4000},
//{19192, 680, 110, 4000},
//{16087, 720, 110, 4000},
//{12969, 760, 110, 4000},
//{10438, 800, 110, 4000},
//{8172, 840, 110, 4000},
//{5045, 880, 110, 4000},
//{2513, 920, 110, 4000},
//{605, 960, 110, 4000},
//{32765, 0, 115, 4000},
//{33038, 40, 115, 4000},
//{33013, 80, 115, 4000},
//{32440, 120, 115, 4000},
//{31593, 160, 115, 4000},
//{31444, 200, 115, 4000},
//{31444, 240, 115, 4000},
//{31235, 280, 115, 4000},
//{30265, 320, 115, 4000},
//{29243, 360, 115, 4000},
//{29035, 400, 115, 4000},
//{28579, 440, 115, 4000},
//{27928, 480, 115, 4000},
//{26274, 520, 115, 4000},
//{24568, 560, 115, 4000},
//{22960, 600, 115, 4000},
//{21743, 640, 115, 4000},
//{19972, 680, 115, 4000},
//{17707, 720, 115, 4000},
//{14634, 760, 115, 4000},
//{11717, 800, 115, 4000},
//{8586, 840, 115, 4000},
//{6040, 880, 115, 4000},
//{3749, 920, 115, 4000},
//{1867, 960, 115, 4000},
//{34171, 0, 120, 4000},
//{33248, 40, 120, 4000},
//{32916, 80, 120, 4000},
//{33066, 120, 120, 4000},
//{32969, 160, 120, 4000},
//{32323, 200, 120, 4000},
//{31901, 240, 120, 4000},
//{31776, 280, 120, 4000},
//{31932, 320, 120, 4000},
//{31580, 360, 120, 4000},
//{29759, 400, 120, 4000},
//{29077, 440, 120, 4000},
//{27950, 480, 120, 4000},
//{27637, 520, 120, 4000},
//{25709, 560, 120, 4000},
//{23834, 600, 120, 4000},
//{22322, 640, 120, 4000},
//{20388, 680, 120, 4000},
//{18175, 720, 120, 4000},
//{15142, 760, 120, 4000},
//{12278, 800, 120, 4000},
//{9471, 840, 120, 4000},
//{6984, 880, 120, 4000},
//{4640, 920, 120, 4000},
//{2362, 960, 120, 4000},
//{32765, 0, 130, 4000},
//{32765, 40, 130, 4000},
//{32622, 80, 130, 4000},
//{32244, 120, 130, 4000},
//{31763, 160, 130, 4000},
//{31249, 200, 130, 4000},
//{30553, 240, 130, 4000},
//{29629, 280, 130, 4000},
//{29062, 320, 130, 4000},
//{28794, 360, 130, 4000},
//{28585, 400, 130, 4000},
//{27778, 440, 130, 4000},
//{26463, 480, 130, 4000},
//{24966, 520, 130, 4000},
//{23411, 560, 130, 4000},
//{22420, 600, 130, 4000},
//{20878, 640, 130, 4000},
//{18827, 680, 130, 4000},
//{15982, 720, 130, 4000},
//{13273, 760, 130, 4000},
//{10487, 800, 130, 4000},
//{7955, 840, 130, 4000},
//{6021, 880, 130, 4000},
//{4524, 920, 130, 4000},
//{2864, 960, 130, 4000},
//{31926, 0, 140, 4000},
//{31405, 40, 140, 4000},
//{30917, 80, 140, 4000},
//{30312, 120, 140, 4000},
//{29876, 160, 140, 4000},
//{29433, 200, 140, 4000},
//{29055, 240, 140, 4000},
//{28579, 280, 140, 4000},
//{28058, 320, 140, 4000},
//{27524, 360, 140, 4000},
//{26952, 400, 140, 4000},
//{26073, 440, 140, 4000},
//{24790, 480, 140, 4000},
//{23332, 520, 140, 4000},
//{22290, 560, 140, 4000},
//{21386, 600, 140, 4000},
//{20128, 640, 140, 4000},
//{17993, 680, 140, 4000},
//{15493, 720, 140, 4000},
//{12941, 760, 140, 4000},
//{10454, 800, 140, 4000},
//{7915, 840, 140, 4000},
//{4605, 880, 140, 4000},
//{1955, 920, 140, 4000},
//{151, 960, 140, 4000},
//{30538, 0, 150, 4000},
//{29758, 40, 150, 4000},
//{29140, 80, 150, 4000},
//{28560, 120, 150, 4000},
//{27948, 160, 150, 4000},
//{27336, 200, 150, 4000},
//{26984, 240, 150, 4000},
//{26607, 280, 150, 4000},
//{26458, 320, 150, 4000},
//{26295, 360, 150, 4000},
//{26256, 400, 150, 4000},
//{25643, 440, 150, 4000},
//{24537, 480, 150, 4000},
//{23157, 520, 150, 4000},
//{22245, 560, 150, 4000},
//{21144, 600, 150, 4000},
//{19712, 640, 150, 4000},
//{17473, 680, 150, 4000},
//{15058, 720, 150, 4000},
//{12538, 760, 150, 4000},
//{10024, 800, 150, 4000},
//{7544, 840, 150, 4000},
//{5396, 880, 150, 4000},
//{3281, 920, 150, 4000},
//{1581, 960, 150, 4000},
//{22765, 0, 160, 4000},
//{22186, 40, 160, 4000},
//{21873, 80, 160, 4000},
//{21821, 120, 160, 4000},
//{21827, 160, 160, 4000},
//{21827, 200, 160, 4000},
//{21827, 240, 160, 4000},
//{21801, 280, 160, 4000},
//{21217, 320, 160, 4000},
//{20965, 360, 160, 4000},
//{20026, 400, 160, 4000},
//{19770, 440, 160, 4000},
//{18369, 480, 160, 4000},
//{17388, 520, 160, 4000},
//{15949, 560, 160, 4000},
//{15012, 600, 160, 4000},
//{13845, 640, 160, 4000},
//{12595, 680, 160, 4000},
//{11241, 720, 160, 4000},
//{9887, 760, 160, 4000},
//{8306, 800, 160, 4000},
//{6561, 840, 160, 4000},
//{4634, 880, 160, 4000},
//{2800, 920, 160, 4000},
//{1212, 960, 160, 4000},
//{26379, 0, 180, 4000},
//{26815, 40, 180, 4000},
//{26880, 80, 180, 4000},
//{26556, 120, 180, 4000},
//{26235, 160, 180, 4000},
//{25864, 200, 180, 4000},
//{25637, 240, 180, 4000},
//{25246, 280, 180, 4000},
//{24940, 320, 180, 4000},
//{24399, 360, 180, 4000},
//{23846, 400, 180, 4000},
//{23163, 440, 180, 4000},
//{22564, 480, 180, 4000},
//{21874, 520, 180, 4000},
//{20963, 560, 180, 4000},
//{19739, 600, 180, 4000},
//{17562, 640, 180, 4000},
//{14182, 680, 180, 4000},
//{11115, 720, 180, 4000},
//{8364, 760, 180, 4000},
//{6770, 800, 180, 4000},
//{5407, 840, 180, 4000},
//{4314, 880, 180, 4000},
//{3304, 920, 180, 4000},
//{1678, 960, 180, 4000},
//{25989, 0, 200, 4000},
//{25488, 40, 200, 4000},
//{25142, 80, 200, 4000},
//{24654, 120, 200, 4000},
//{24510, 160, 200, 4000},
//{24119, 200, 200, 4000},
//{23787, 240, 200, 4000},
//{23384, 280, 200, 4000},
//{22974, 320, 200, 4000},
//{22558, 360, 200, 4000},
//{22024, 400, 200, 4000},
//{21729, 440, 200, 4000},
//{21020, 480, 200, 4000},
//{19978, 520, 200, 4000},
//{18430, 560, 200, 4000},
//{16893, 600, 200, 4000},
//{15331, 640, 200, 4000},
//{13443, 680, 200, 4000},
//{11502, 720, 200, 4000},
//{9549, 760, 200, 4000},
//{7778, 800, 200, 4000},
//{6007, 840, 200, 4000},
//{4360, 880, 200, 4000},
//{2870, 920, 200, 4000},
//{1486, 960, 200, 4000},
//{23624, 0, 220, 4000},
//{23285, 40, 220, 4000},
//{22934, 80, 220, 4000},
//{22472, 120, 220, 4000},
//{22075, 160, 220, 4000},
//{21827, 200, 220, 4000},
//{21814, 240, 220, 4000},
//{21678, 280, 220, 4000},
//{21327, 320, 220, 4000},
//{20871, 360, 220, 4000},
//{20343, 400, 220, 4000},
//{19595, 440, 220, 4000},
//{18599, 480, 220, 4000},
//{17219, 520, 220, 4000},
//{15890, 560, 220, 4000},
//{14406, 600, 220, 4000},
//{12882, 640, 220, 4000},
//{11105, 680, 220, 4000},
//{9314, 720, 220, 4000},
//{7517, 760, 220, 4000},
//{5818, 800, 220, 4000},
//{4165, 840, 220, 4000},
//{2694, 880, 220, 4000},
//{1382, 920, 220, 4000},
//{497, 960, 220, 4000},
//{21398, 0, 240, 4000},
//{21047, 40, 240, 4000},
//{20754, 80, 240, 4000},
//{20422, 120, 240, 4000},
//{20154, 160, 240, 4000},
//{19861, 200, 240, 4000},
//{19607, 240, 240, 4000},
//{19269, 280, 240, 4000},
//{18892, 320, 240, 4000},
//{18508, 360, 240, 4000},
//{18098, 400, 240, 4000},
//{17518, 440, 240, 4000},
//{16593, 480, 240, 4000},
//{15441, 520, 240, 4000},
//{14230, 560, 240, 4000},
//{12994, 600, 240, 4000},
//{11594, 640, 240, 4000},
//{9999, 680, 240, 4000},
//{8320, 720, 240, 4000},
//{6698, 760, 240, 4000},
//{5188, 800, 240, 4000},
//{3827, 840, 240, 4000},
//{2532, 880, 240, 4000},
//{1271, 920, 240, 4000},
//{432, 960, 240, 4000},
//{21730, 0, 260, 4000},
//{21210, 40, 260, 4000},
//{20839, 80, 260, 4000},
//{20519, 120, 260, 4000},
//{20083, 160, 260, 4000},
//{19627, 200, 260, 4000},
//{19223, 240, 260, 4000},
//{18819, 280, 260, 4000},
//{18383, 320, 260, 4000},
//{18436, 360, 260, 4000},
//{18449, 400, 260, 4000},
//{18221, 440, 260, 4000},
//{17270, 480, 260, 4000},
//{16079, 520, 260, 4000},
//{14790, 560, 260, 4000},
//{13515, 600, 260, 4000},
//{12141, 640, 260, 4000},
//{10780, 680, 260, 4000},
//{9257, 720, 260, 4000},
//{7811, 760, 260, 4000},
//{6313, 800, 260, 4000},
//{4894, 840, 260, 4000},
//{3631, 880, 260, 4000},
//{2545, 920, 260, 4000},
//{1614, 960, 260, 4000},
//{21262, 0, 280, 4000},
//{20890, 40, 280, 4000},
//{20545, 80, 280, 4000},
//{20129, 120, 280, 4000},
//{19771, 160, 280, 4000},
//{19367, 200, 280, 4000},
//{18785, 240, 280, 4000},
//{18355, 280, 280, 4000},
//{17965, 320, 280, 4000},
//{17739, 360, 280, 4000},
//{17310, 400, 280, 4000},
//{16619, 440, 280, 4000},
//{15714, 480, 280, 4000},
//{14588, 520, 280, 4000},
//{13403, 560, 280, 4000},
//{12238, 600, 280, 4000},
//{10988, 640, 280, 4000},
//{9673, 680, 280, 4000},
//{8293, 720, 280, 4000},
//{6984, 760, 280, 4000},
//{5690, 800, 280, 4000},
//{4491, 840, 280, 4000},
//{3359, 880, 280, 4000},
//{2362, 920, 280, 4000},
//{1388, 960, 280, 4000},
//{13585, 0, 320, 4000},
//{13391, 40, 320, 4000},
//{13202, 80, 320, 4000},
//{12936, 120, 320, 4000},
//{12610, 160, 320, 4000},
//{12252, 200, 320, 4000},
//{11985, 240, 320, 4000},
//{11776, 280, 320, 4000},
//{11567, 320, 320, 4000},
//{11320, 360, 320, 4000},
//{10989, 400, 320, 4000},
//{10585, 440, 320, 4000},
//{9993, 480, 320, 4000},
//{9263, 520, 320, 4000},
//{8462, 560, 320, 4000},
//{7785, 600, 320, 4000},
//{7232, 640, 320, 4000},
//{6685, 680, 320, 4000},
//{6047, 720, 320, 4000},
//{5304, 760, 320, 4000},
//{4503, 800, 320, 4000},
//{3598, 840, 320, 4000},
//{2648, 880, 320, 4000},
//{1737, 920, 320, 4000},
//{881, 960, 320, 4000},
//{16086, 0, 360, 4000},
//{15669, 40, 360, 4000},
//{15330, 80, 360, 4000},
//{15096, 120, 360, 4000},
//{14796, 160, 360, 4000},
//{14465, 200, 360, 4000},
//{14075, 240, 360, 4000},
//{13743, 280, 360, 4000},
//{13455, 320, 360, 4000},
//{13150, 360, 360, 4000},
//{12786, 400, 360, 4000},
//{12284, 440, 360, 4000},
//{11554, 480, 360, 4000},
//{10630, 520, 360, 4000},
//{9686, 560, 360, 4000},
//{8827, 600, 360, 4000},
//{7960, 640, 360, 4000},
//{6945, 680, 360, 4000},
//{5904, 720, 360, 4000},
//{4824, 760, 360, 4000},
//{3840, 800, 360, 4000},
//{2922, 840, 360, 4000},
//{2108, 880, 360, 4000},
//{1386, 920, 360, 4000},
//{682, 960, 360, 4000},
//{13802, 0, 400, 4000},
//{13579, 40, 400, 4000},
//{13338, 80, 400, 4000},
//{13098, 120, 400, 4000},
//{12799, 160, 400, 4000},
//{12500, 200, 400, 4000},
//{12180, 240, 400, 4000},
//{11900, 280, 400, 4000},
//{11613, 320, 400, 4000},
//{11327, 360, 400, 4000},
//{11021, 400, 400, 4000},
//{10558, 440, 400, 4000},
//{9920, 480, 400, 4000},
//{9061, 520, 400, 4000},
//{8260, 560, 400, 4000},
//{7551, 600, 400, 4000},
//{6847, 640, 400, 4000},
//{5988, 680, 400, 4000},
//{4992, 720, 400, 4000},
//{4048, 760, 400, 4000},
//{3188, 800, 400, 4000},
//{2406, 840, 400, 4000},
//{1684, 880, 400, 4000},
//{1027, 920, 400, 4000},
//{480, 960, 400, 4000},
//{12200, 0, 440, 4000},
//{11984, 40, 440, 4000},
//{11737, 80, 440, 4000},
//{11483, 120, 440, 4000},
//{11229, 160, 440, 4000},
//{10949, 200, 440, 4000},
//{10631, 240, 440, 4000},
//{10365, 280, 440, 4000},
//{10156, 320, 440, 4000},
//{9946, 360, 440, 4000},
//{9646, 400, 440, 4000},
//{9224, 440, 440, 4000},
//{8657, 480, 440, 4000},
//{7941, 520, 440, 4000},
//{7270, 560, 440, 4000},
//{6593, 600, 440, 4000},
//{5878, 640, 440, 4000},
//{5039, 680, 440, 4000},
//{4200, 720, 440, 4000},
//{3424, 760, 440, 4000},
//{2700, 800, 440, 4000},
//{2017, 840, 440, 4000},
//{1385, 880, 440, 4000},
//{838, 920, 440, 4000},
//{382, 960, 440, 4000},
//{10206, 0, 500, 4000},
//{10018, 40, 500, 4000},
//{9829, 80, 500, 4000},
//{9608, 120, 500, 4000},
//{9386, 160, 500, 4000},
//{9152, 200, 500, 4000},
//{8931, 240, 500, 4000},
//{8697, 280, 500, 4000},
//{8476, 320, 500, 4000},
//{8261, 360, 500, 4000},
//{8027, 400, 500, 4000},
//{7669, 440, 500, 4000},
//{7200, 480, 500, 4000},
//{6608, 520, 500, 4000},
//{5989, 560, 500, 4000},
//{5357, 600, 500, 4000},
//{4699, 640, 500, 4000},
//{4042, 680, 500, 4000},
//{3398, 720, 500, 4000},
//{2778, 760, 500, 4000},
//{2160, 800, 500, 4000},
//{1554, 840, 500, 4000},
//{1036, 880, 500, 4000},
//{538, 920, 500, 4000},
//    {200, 960, 500, 4000}};

  /* THIS IS FOR N = 500 */
  int hc_settings[N_SETTINGS][4] = {
  {19432, 0, 100, 4000},
{17976, 20, 100, 4000},
{17413, 40, 100, 4000},
{17524, 60, 100, 4000},
{17159, 80, 100, 4000},
{16432, 100, 100, 4000},
{15611, 120, 100, 4000},
{14817, 140, 100, 4000},
{13937, 160, 100, 4000},
{13338, 180, 100, 4000},
{12623, 200, 100, 4000},
{12284, 220, 100, 4000},
{11594, 240, 100, 4000},
{11080, 260, 100, 4000},
{10267, 280, 100, 4000},
{9648, 300, 100, 4000},
{8933, 320, 100, 4000},
{8340, 340, 100, 4000},
{7630, 360, 100, 4000},
{6983, 380, 100, 4000},
{6346, 400, 100, 4000},
{5709, 420, 100, 4000},
{4680, 440, 100, 4000},
{3456, 460, 100, 4000},
{1792, 480, 100, 4000},
{18429, 0, 105, 4000},
{17238, 20, 105, 4000},
{16495, 40, 105, 4000},
{15845, 60, 105, 4000},
{15584, 80, 105, 4000},
{15136, 100, 105, 4000},
{14420, 120, 105, 4000},
{13951, 140, 105, 4000},
{13261, 160, 105, 4000},
{12909, 180, 105, 4000},
{12408, 200, 105, 4000},
{11978, 220, 105, 4000},
{11398, 240, 105, 4000},
{10838, 260, 105, 4000},
{9959, 280, 105, 4000},
{9374, 300, 105, 4000},
{8715, 320, 105, 4000},
{8337, 340, 105, 4000},
{7706, 360, 105, 4000},
{7121, 380, 105, 4000},
{6425, 400, 105, 4000},
{5650, 420, 105, 4000},
{4310, 440, 105, 4000},
{3021, 460, 105, 4000},
{1423, 480, 105, 4000},
{17632, 0, 110, 4000},
{16920, 20, 110, 4000},
{16340, 40, 110, 4000},
{15910, 60, 110, 4000},
{15402, 80, 110, 4000},
{14849, 100, 110, 4000},
{14244, 120, 110, 4000},
{13651, 140, 110, 4000},
{13058, 160, 110, 4000},
{12486, 180, 110, 4000},
{11900, 200, 110, 4000},
{11366, 220, 110, 4000},
{10819, 240, 110, 4000},
{10260, 260, 110, 4000},
{9596, 280, 110, 4000},
{9017, 300, 110, 4000},
{8371, 320, 110, 4000},
{7967, 340, 110, 4000},
{7381, 360, 110, 4000},
{6867, 380, 110, 4000},
{6177, 400, 110, 4000},
{5520, 420, 110, 4000},
{4648, 440, 110, 4000},
{3398, 460, 110, 4000},
{1789, 480, 110, 4000},
{16904, 0, 115, 4000},
{16411, 20, 115, 4000},
{15943, 40, 115, 4000},
{15500, 60, 115, 4000},
{15057, 80, 115, 4000},
{14601, 100, 115, 4000},
{14010, 120, 115, 4000},
{13366, 140, 115, 4000},
{12650, 160, 115, 4000},
{12082, 180, 115, 4000},
{11554, 200, 115, 4000},
{11138, 220, 115, 4000},
{10695, 240, 115, 4000},
{10200, 260, 115, 4000},
{9588, 280, 115, 4000},
{8970, 300, 115, 4000},
{8449, 320, 115, 4000},
{7889, 340, 115, 4000},
{7336, 360, 115, 4000},
{6686, 380, 115, 4000},
{6133, 400, 115, 4000},
{5494, 420, 115, 4000},
{4497, 440, 115, 4000},
{3266, 460, 115, 4000},
{1665, 480, 115, 4000},
{16498, 0, 120, 4000},
{16055, 20, 120, 4000},
{15631, 40, 120, 4000},
{15194, 60, 120, 4000},
{14751, 80, 120, 4000},
{14295, 100, 120, 4000},
{13624, 120, 120, 4000},
{12993, 140, 120, 4000},
{12343, 160, 120, 4000},
{11848, 180, 120, 4000},
{11333, 200, 120, 4000},
{10760, 220, 120, 4000},
{10213, 240, 120, 4000},
{9692, 260, 120, 4000},
{9204, 280, 120, 4000},
{8723, 300, 120, 4000},
{8189, 320, 120, 4000},
{7648, 340, 120, 4000},
{7049, 360, 120, 4000},
{6516, 380, 120, 4000},
{5917, 400, 120, 4000},
{5214, 420, 120, 4000},
{4328, 440, 120, 4000},
{3116, 460, 120, 4000},
{1658, 480, 120, 4000},
{14992, 0, 130, 4000},
{14608, 20, 130, 4000},
{14224, 40, 130, 4000},
{13891, 60, 130, 4000},
{13488, 80, 130, 4000},
{13065, 100, 130, 4000},
{12538, 120, 130, 4000},
{11940, 140, 130, 4000},
{11386, 160, 130, 4000},
{10872, 180, 130, 4000},
{10390, 200, 130, 4000},
{9856, 220, 130, 4000},
{9420, 240, 130, 4000},
{8873, 260, 130, 4000},
{8358, 280, 130, 4000},
{7739, 300, 130, 4000},
{7264, 320, 130, 4000},
{6789, 340, 130, 4000},
{6366, 360, 130, 4000},
{5903, 380, 130, 4000},
{5448, 400, 130, 4000},
{4797, 420, 130, 4000},
{3938, 440, 130, 4000},
{2830, 460, 130, 4000},
{1496, 480, 130, 4000},
{14152, 0, 140, 4000},
{13710, 20, 140, 4000},
{13286, 40, 140, 4000},
{12941, 60, 140, 4000},
{12505, 80, 140, 4000},
{12075, 100, 140, 4000},
{11554, 120, 140, 4000},
{11098, 140, 140, 4000},
{10604, 160, 140, 4000},
{10200, 180, 140, 4000},
{9692, 200, 140, 4000},
{9210, 220, 140, 4000},
{8748, 240, 140, 4000},
{8305, 260, 140, 4000},
{7850, 280, 140, 4000},
{7264, 300, 140, 4000},
{6750, 320, 140, 4000},
{6307, 340, 140, 4000},
{5884, 360, 140, 4000},
{5447, 380, 140, 4000},
{4952, 400, 140, 4000},
{4458, 420, 140, 4000},
{3729, 440, 140, 4000},
{2668, 460, 140, 4000},
{1366, 480, 140, 4000},
{13138, 0, 150, 4000},
{12792, 20, 150, 4000},
{12551, 40, 150, 4000},
{12212, 60, 150, 4000},
{11828, 80, 150, 4000},
{11469, 100, 150, 4000},
{11073, 120, 150, 4000},
{10643, 140, 150, 4000},
{10025, 160, 150, 4000},
{9536, 180, 150, 4000},
{9159, 200, 150, 4000},
{8767, 220, 150, 4000},
{8338, 240, 150, 4000},
{7843, 260, 150, 4000},
{7453, 280, 150, 4000},
{7062, 300, 150, 4000},
{6593, 320, 150, 4000},
{6072, 340, 150, 4000},
{5682, 360, 150, 4000},
{5226, 380, 150, 4000},
{4849, 400, 150, 4000},
{4354, 420, 150, 4000},
{3742, 440, 150, 4000},
{2759, 460, 150, 4000},
{1447, 480, 150, 4000},
{11320, 0, 160, 4000},
{11060, 20, 160, 4000},
{10787, 40, 160, 4000},
{10513, 60, 160, 4000},
{10083, 80, 160, 4000},
{9712, 100, 160, 4000},
{9485, 120, 160, 4000},
{9276, 140, 160, 4000},
{9037, 160, 160, 4000},
{8743, 180, 160, 4000},
{8437, 200, 160, 4000},
{8110, 220, 160, 4000},
{7680, 240, 160, 4000},
{7257, 260, 160, 4000},
{6867, 280, 160, 4000},
{6483, 300, 160, 4000},
{6053, 320, 160, 4000},
{5643, 340, 160, 4000},
{5220, 360, 160, 4000},
{4803, 380, 160, 4000},
{4379, 400, 160, 4000},
{3892, 420, 160, 4000},
{3241, 440, 160, 4000},
{2303, 460, 160, 4000},
{1189, 480, 160, 4000},
{11125, 0, 180, 4000},
{10799, 20, 180, 4000},
{10526, 40, 180, 4000},
{10213, 60, 180, 4000},
{9920, 80, 180, 4000},
{9484, 100, 180, 4000},
{9016, 120, 180, 4000},
{8553, 140, 180, 4000},
{8156, 160, 180, 4000},
{7843, 180, 180, 4000},
{7576, 200, 180, 4000},
{7297, 220, 180, 4000},
{6985, 240, 180, 4000},
{6659, 260, 180, 4000},
{6346, 280, 180, 4000},
{5994, 300, 180, 4000},
{5604, 320, 180, 4000},
{5161, 340, 180, 4000},
{4751, 360, 180, 4000},
{4335, 380, 180, 4000},
{3963, 400, 180, 4000},
{3410, 420, 180, 4000},
{2720, 440, 180, 4000},
{1802, 460, 180, 4000},
{891, 480, 180, 4000},
{9699, 0, 200, 4000},
{9426, 20, 200, 4000},
{9159, 40, 200, 4000},
{8892, 60, 200, 4000},
{8612, 80, 200, 4000},
{8293, 100, 200, 4000},
{7955, 120, 200, 4000},
{7590, 140, 200, 4000},
{7284, 160, 200, 4000},
{6938, 180, 200, 4000},
{6600, 200, 200, 4000},
{6229, 220, 200, 4000},
{5891, 240, 200, 4000},
{5578, 260, 200, 4000},
{5252, 280, 200, 4000},
{4947, 300, 200, 4000},
{4576, 320, 200, 4000},
{4244, 340, 200, 4000},
{3879, 360, 200, 4000},
{3599, 380, 200, 4000},
{3286, 400, 200, 4000},
{2928, 420, 200, 4000},
{2368, 440, 200, 4000},
{1633, 460, 200, 4000},
{813, 480, 200, 4000},
{8859, 0, 220, 4000},
{8637, 20, 220, 4000},
{8397, 40, 220, 4000},
{8130, 60, 220, 4000},
{7844, 80, 220, 4000},
{7563, 100, 220, 4000},
{7277, 120, 220, 4000},
{6932, 140, 220, 4000},
{6601, 160, 220, 4000},
{6255, 180, 220, 4000},
{5969, 200, 220, 4000},
{5682, 220, 220, 4000},
{5383, 240, 220, 4000},
{5057, 260, 220, 4000},
{4732, 280, 220, 4000},
{4439, 300, 220, 4000},
{4166, 320, 220, 4000},
{3840, 340, 220, 4000},
{3534, 360, 220, 4000},
{3214, 380, 220, 4000},
{2889, 400, 220, 4000},
{2512, 420, 220, 4000},
{2030, 440, 220, 4000},
{1444, 460, 220, 4000},
{741, 480, 220, 4000},
{7980, 0, 240, 4000},
{7766, 20, 240, 4000},
{7538, 40, 240, 4000},
{7323, 60, 240, 4000},
{7115, 80, 240, 4000},
{6848, 100, 240, 4000},
{6561, 120, 240, 4000},
{6196, 140, 240, 4000},
{5897, 160, 240, 4000},
{5604, 180, 240, 4000},
{5331, 200, 240, 4000},
{5064, 220, 240, 4000},
{4784, 240, 240, 4000},
{4491, 260, 240, 4000},
{4237, 280, 240, 4000},
{3969, 300, 240, 4000},
{3729, 320, 240, 4000},
{3436, 340, 240, 4000},
{3143, 360, 240, 4000},
{2850, 380, 240, 4000},
{2563, 400, 240, 4000},
{2277, 420, 240, 4000},
{1854, 440, 240, 4000},
{1294, 460, 240, 4000},
{636, 480, 240, 4000},
{7414, 0, 260, 4000},
{7180, 20, 260, 4000},
{6939, 40, 260, 4000},
{6712, 60, 260, 4000},
{6471, 80, 260, 4000},
{6262, 100, 260, 4000},
{6008, 120, 260, 4000},
{5734, 140, 260, 4000},
{5402, 160, 260, 4000},
{5142, 180, 260, 4000},
{4920, 200, 260, 4000},
{4712, 220, 260, 4000},
{4458, 240, 260, 4000},
{4178, 260, 260, 4000},
{3943, 280, 260, 4000},
{3696, 300, 260, 4000},
{3449, 320, 260, 4000},
{3189, 340, 260, 4000},
{2949, 360, 260, 4000},
{2701, 380, 260, 4000},
{2434, 400, 260, 4000},
{2121, 420, 260, 4000},
{1698, 440, 260, 4000},
{1157, 460, 260, 4000},
{565, 480, 260, 4000},
{6652, 0, 280, 4000},
{6503, 20, 280, 4000},
{6301, 40, 280, 4000},
{6093, 60, 280, 4000},
{5884, 80, 280, 4000},
{5689, 100, 280, 4000},
{5441, 120, 280, 4000},
{5193, 140, 280, 4000},
{4934, 160, 280, 4000},
{4693, 180, 280, 4000},
{4452, 200, 280, 4000},
{4217, 220, 280, 4000},
{3996, 240, 280, 4000},
{3775, 260, 280, 4000},
{3567, 280, 280, 4000},
{3326, 300, 280, 4000},
{3098, 320, 280, 4000},
{2850, 340, 280, 4000},
{2616, 360, 280, 4000},
{2382, 380, 280, 4000},
{2134, 400, 280, 4000},
{1880, 420, 280, 4000},
{1502, 440, 280, 4000},
{1033, 460, 280, 4000},
{499, 480, 280, 4000},
{5226, 0, 320, 4000},
{5142, 20, 320, 4000},
{5038, 40, 320, 4000},
{4933, 60, 320, 4000},
{4797, 80, 320, 4000},
{4654, 100, 320, 4000},
{4498, 120, 320, 4000},
{4335, 140, 320, 4000},
{4172, 160, 320, 4000},
{4002, 180, 320, 4000},
{3821, 200, 320, 4000},
{3580, 220, 320, 4000},
{3365, 240, 320, 4000},
{3157, 260, 320, 4000},
{2987, 280, 320, 4000},
{2785, 300, 320, 4000},
{2596, 320, 320, 4000},
{2414, 340, 320, 4000},
{2212, 360, 320, 4000},
{1978, 380, 320, 4000},
{1757, 400, 320, 4000},
{1528, 420, 320, 4000},
{1242, 440, 320, 4000},
{812, 460, 320, 4000},
{376, 480, 320, 4000},
{4874, 0, 360, 4000},
{4718, 20, 360, 4000},
{4556, 40, 360, 4000},
{4380, 60, 360, 4000},
{4211, 80, 360, 4000},
{4062, 100, 360, 4000},
{3906, 120, 360, 4000},
{3723, 140, 360, 4000},
{3527, 160, 360, 4000},
{3358, 180, 360, 4000},
{3202, 200, 360, 4000},
{3032, 220, 360, 4000},
{2863, 240, 360, 4000},
{2720, 260, 360, 4000},
{2571, 280, 360, 4000},
{2388, 300, 360, 4000},
{2193, 320, 360, 4000},
{2004, 340, 360, 4000},
{1847, 360, 360, 4000},
{1678, 380, 360, 4000},
{1509, 400, 360, 4000},
{1294, 420, 360, 4000},
{1047, 440, 360, 4000},
{688, 460, 360, 4000},
{337, 480, 360, 4000},
{4153, 0, 400, 4000},
{4042, 20, 400, 4000},
{3911, 40, 400, 4000},
{3787, 60, 400, 4000},
{3657, 80, 400, 4000},
{3533, 100, 400, 4000},
{3364, 120, 400, 4000},
{3189, 140, 400, 4000},
{3006, 160, 400, 4000},
{2889, 180, 400, 4000},
{2752, 200, 400, 4000},
{2635, 220, 400, 4000},
{2465, 240, 400, 4000},
{2316, 260, 400, 4000},
{2166, 280, 400, 4000},
{2024, 300, 400, 4000},
{1874, 320, 400, 4000},
{1711, 340, 400, 4000},
{1561, 360, 400, 4000},
{1418, 380, 400, 4000},
{1262, 400, 400, 4000},
{1093, 420, 400, 4000},
{892, 440, 400, 4000},
{598, 460, 400, 4000},
{292, 480, 400, 4000},
{3664, 0, 440, 4000},
{3541, 20, 440, 4000},
{3437, 40, 440, 4000},
{3332, 60, 440, 4000},
{3221, 80, 440, 4000},
{3091, 100, 440, 4000},
{2948, 120, 440, 4000},
{2805, 140, 440, 4000},
{2674, 160, 440, 4000},
{2544, 180, 440, 4000},
{2420, 200, 440, 4000},
{2303, 220, 440, 4000},
{2192, 240, 440, 4000},
{2062, 260, 440, 4000},
{1919, 280, 440, 4000},
{1782, 300, 440, 4000},
{1652, 320, 440, 4000},
{1522, 340, 440, 4000},
{1398, 360, 440, 4000},
{1255, 380, 440, 4000},
{1105, 400, 440, 4000},
{923, 420, 440, 4000},
{766, 440, 440, 4000},
{512, 460, 440, 4000},
{265, 480, 440, 4000},
{3077, 0, 500, 4000},
{2974, 20, 500, 4000},
{2948, 40, 500, 4000},
{2877, 60, 500, 4000},
{2799, 80, 500, 4000},
{2629, 100, 500, 4000},
{2498, 120, 500, 4000},
{2446, 140, 500, 4000},
{2388, 160, 500, 4000},
{2349, 180, 500, 4000},
{2161, 200, 500, 4000},
{1991, 220, 500, 4000},
{1815, 240, 500, 4000},
{1704, 260, 500, 4000},
{1593, 280, 500, 4000},
{1470, 300, 500, 4000},
{1354, 320, 500, 4000},
{1236, 340, 500, 4000},
{1125, 360, 500, 4000},
{1001, 380, 500, 4000},
{871, 400, 500, 4000},
{741, 420, 500, 4000},
{611, 440, 500, 4000},
{379, 460, 500, 4000},
    {171, 480, 500, 4000}};
  
  for(int i = 0; i < N_SETTINGS; i++)
    for(int j = 0; j < 4; j++)
      settings[i][j] = hc_settings[i][j];
  
}

