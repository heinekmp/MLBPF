/*
 * particle_filters.h
 *
 *  Created on: 18 Sep 2020
 *      Author: heine
 */

#ifndef PARTICLE_FILTERS_H_
#define PARTICLE_FILTERS_H_

double likelihood(int n, double x, double *y, int Nobs0, double *Rinv, int Nobs, double det);
double full_likelihood(int n, double x, double *y, double *Rinv, int Nobs, double det);
void MLbootstrapfilter(double* y, double sig_std, double obs_std, int data_length,
    double x0, int Nobs, short N_levels, long* sample_sizes, double* x_hats,
    double *filtering_time, double *Rinv0, double *Rinv1, double *worst_case_sign_ratio);
void resample(long size, double *w, long *ind, gsl_rng *r);
void random_permuter(long *permutation, long N, gsl_rng *r);
void bootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
     int Nobs, long N, double* x_hats, double *filtering_time, double *Rinv,
    short full);


#endif /* PARTICLE_FILTERS_H_ */
