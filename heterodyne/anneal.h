#ifndef ANNEAL_H_INCLUDED
#define ANNEAL_H_INCLUDED
#include <complex.h>

// returns a nsamples by n array of complex numbers.
void sample_heterodyne(int n, int *f, complex double **U, complex double **P, double *ls, int nsamples, int initial_anneal, int Delta, complex double *initial_alpha, complex double **result);

#endif