/*
 * kalman_filter.c
 *
 *  Created on: 18 Sep 2020
 *      Author: heine
 */
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include "kalman_filter.h"

/*
 * Kalman filter for one dimensional signal, but high dimensional observations
 */
void kalman_filter(double* x, double* y, double* Rinv, double sig_std, double obs_std,
    double *x_hat, double *p_hat, int data_length, double x0, double p0, int Nobs) {

  const char prog_bar[51] = "-------------------------------------------------";

  clock_t start = clock();
  double* innovation = (double*) malloc(Nobs * sizeof(double));
  double * Rinv2 = (double*) malloc(Nobs * Nobs * sizeof(double));
  double * Sinv = (double*) malloc(Nobs * Nobs * sizeof(double));
  double KH;
  double x_diff;
  double Rfrob = 0;
  double tmp;
  int bar_segment_count = 0;

  /* Initialise */
  x_hat[0] = x0;
  p_hat[0] = p0;

  /* In one dimensional case, we end up needing the Frobenius norm of observation
   * covariance R (Rfrob) as well as R^{-1}*ones * ones^T R^{-1}, which we call Rinv2
   * TODO:(double check this formula) */
  double tmp1, tmp2;
  for (int i = 0; i < Nobs; i++) {

    tmp1 = 0;
    for (int k = 0; k < Nobs; k++) {
      tmp1 += Rinv[i * Nobs + k];
      Rfrob += Rinv[i * Nobs + k];
    }

    for (int j = 0; j < Nobs; j++) {
      tmp2 = 0;
      for (int k = 0; k < Nobs; k++)
        tmp2 += Rinv[k * Nobs + j];

      Rinv2[i * Nobs + j] = tmp1 * tmp2;
    }
  }

  printf("\n");
  printf(
      "%s\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
      prog_bar);
  fflush(stdout);

  start = clock();
  for (int n = 0; n < data_length; n++) {

    /* Innovation */
    for (int i = 0; i < Nobs; i++) {
      innovation[i] = y[n * Nobs + i] - x_hat[n];
    }

    /* Inverse of S using Woodbury matrix identity */
    for (int i = 0; i < Nobs; i++) {
      for (int j = 0; j < Nobs; j++) {
        Sinv[i * Nobs + j] = Rinv[i * Nobs + j]
            - (double) 1.0 / ((double) 1.0 / p_hat[n] + Rfrob) * Rinv2[i * Nobs + j];
      }
    }

    /* Compute the Kalman gain */
    KH = 0;
    x_diff = 0;
    for (int i = 0; i < Nobs; i++) {
      tmp = 0;
      for (int j = 0; j < Nobs; j++) {
        tmp += Sinv[i * Nobs + j];
      }
      tmp *= p_hat[n];
      KH += tmp;
      x_diff += tmp * innovation[i];
    }

    /* Update the variance */
    p_hat[n] *= ((double) 1.0 - KH);

    /* Update the state estimate */
    x_hat[n] += x_diff;

    /* Prediction */
    if (n < data_length - 1) {
      x_hat[n + 1] = x_hat[n];
      p_hat[n + 1] = p_hat[n] + sig_std * sig_std;
    }
    if (floor(n * (double) 50 / data_length) > bar_segment_count) {
      printf("█");
      fflush(stdout);
      bar_segment_count++;
    }
  }
  printf(" %5.2f sec\n", (double) (clock() - start) / CLOCKS_PER_SEC);

}
