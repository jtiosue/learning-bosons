#ifndef HOMODYNE_OVERLAP_H
#define HOMODYNE_OVERLAP_H

#include <complex.h>

complex double matelem(
    int l,
    int *m,
    int *n,
    complex double **U,
    complex double **p,
    double *ls,
    complex double *alpha
);

#endif