/*
 * kalman_filter.h
 *
 *  Created on: 18 Sep 2020
 *      Author: heine
 */

#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

void kalman_filter(double* x, double* y, double* Rinv, double sig_std, double obs_std,
    double *x_hat, double *p_hat, int data_length, double x0, double p0, int Nobs);

#endif /* KALMAN_FILTER_H_ */
