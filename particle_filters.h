/*
 * particle_filters.h
 *
 *  Created on: 18 Sep 2020
 *      Author: heine
 */

#ifndef PARTICLE_FILTERS_H_
#define PARTICLE_FILTERS_H_

void likelihood(double* x, int N, double* y, double std, int mesh_size, int kl, int ku, double* ab, int ldab, int* ipiv, int ldb, int info, double* b, double w, double d, double h, double E, double I, double* likes, double L, int* minds, int N_meas, int dodb, double* intercepts, double* slopes);
void MLbootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
                       int* sample_sizes, double *worst_case_sign_ratio, double* x_hats, double *filtering_time, int* mesh_sizes,  int kl, int ku, double* ab0, double* ab1, int ldab, int* ipiv0, int* ipiv1, int ldb0, int ldb1, int info, double w, double d, double E, double I, double h0, double h1, double L , double* meass, int N_meas);
void resample(int size, double *w, int *ind, gsl_rng *r);
void random_permuter(int *permutation, int N, gsl_rng *r);
void bootstrapfilter(double* y, double sig_std, double obs_std, int data_length, double x0,
                     int N, double* x_hats, double* p_hats, double *filtering_time, int mesh_size,  int kl, int ku, double* ab, int ldab, int* ipiv, int ldb, int info, double w, double d, double E, double I, double h , double L, double *meass, int n_meas);

void create_payload(int N, int n, double *b, double w, double d, double* x, double h, double E, double I, double L);
void ebb_solve(int N, double* x, int mesh_size, double* b, double w, double d, double h, double E, double I, double L, int kl, int ku, double *ab, int ldab, int *ipiv, int ldb, int info, int N_meas, int* minds, double* pred_measurements);
void array_copy(double *src, double* dst, int length);
void ls_fit(int N_meas, int N1, double* X1, double* pred_meass0, double* pred_meass1, double* intercepts, double* slopes);

#endif /* PARTICLE_FILTERS_H_ */
