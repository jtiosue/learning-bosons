#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "overlap.h"
#include "random.h"

double prob(int n, int *f, int *vac, complex double **U, complex double **P, double *ls, complex double *alpha, complex double *buf)
{
    for (int i = 0; i < n; ++i)
    {
        // buf[i] = -alpha[i];
        buf[i] = -conj(alpha[i]);
    }

    // return pow(cabs(matelem(n, vac, f, U, P, ls, buf)), 2);
    double inner;
    overlap(n, vac, f, U, P, ls, buf, &inner);
    return pow(inner, 2);
}

void copy_array(int n, complex double *dest, complex double *src)
{
    for (int i = 0; i < n; ++i)
    {
        dest[i] = src[i];
    }
}

void sample_heterodyne(int n, int *f, complex double **U, complex double **P, double *ls, int nsamples, double stepsize, int initial_anneal, int Delta, complex double *initial_alpha, complex double **result)
{
    int *vac = calloc((size_t)n, sizeof(int));
    complex double *buf = malloc((size_t)n * sizeof(complex double));
#define probability(alpha) (prob(n, f, vac, U, P, ls, alpha, buf))
#define copy(a, b) (copy_array(n, a, b))

    rng_t rng = rand_init(-1);

    // heuristic
    // double variance = 0;
    // for (int i = 0; i < n; ++i) {
    //     variance += exp(ls[i]) * (1 + 2 * f[i]);
    // }
    // variance /= (4*n);

    complex double *alpha = malloc((size_t)n * sizeof(complex double));
    complex double *newalpha = malloc((size_t)n * sizeof(complex double));
    copy(alpha, initial_alpha);
    double current_pr = probability(alpha);
    double new_pr;
    int saved = 0;
    for (int step = 0; step < initial_anneal + Delta * nsamples; ++step)
    {
        for (int i = 0; i < n; ++i)
        {
            newalpha[i] = alpha[i] + ((2 * rand_double(&rng) - 1) + I * (2 * rand_double(&rng) - 1)) * stepsize;
        }
        new_pr = probability(newalpha);

        if (current_pr == 0 || (rand_double(&rng) < new_pr / current_pr))
        {
            copy(alpha, newalpha);
            current_pr = new_pr;
        }

        if (step >= initial_anneal && !((step - initial_anneal) % Delta))
        {
            copy(result[saved], alpha);
            saved += 1;
        }
    }
    free(alpha);
    free(newalpha);
    free(vac);
    free(buf);
}