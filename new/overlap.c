#include "overlap.h"

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IDX(i, j, n) ((i) * (n) + (j))

typedef struct {
    int n;
    complex double *data;
} ComplexMatrix;

typedef struct {
    int n;
    double *data;
} RealMatrix;

static complex double invalid_result(void) {
    return CMPLX(NAN, NAN);
}

static ComplexMatrix make_complex_matrix(int n) {
    ComplexMatrix matrix;
    matrix.n = n;
    matrix.data = calloc((size_t) n * (size_t) n, sizeof(complex double));
    return matrix;
}

static RealMatrix make_real_matrix(int n) {
    RealMatrix matrix;
    matrix.n = n;
    matrix.data = calloc((size_t) n * (size_t) n, sizeof(double));
    return matrix;
}

static void free_complex_matrix(ComplexMatrix *matrix) {
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->n = 0;
}

static void free_real_matrix(RealMatrix *matrix) {
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->n = 0;
}

static bool matrix_ok(ComplexMatrix matrix) {
    return matrix.data != NULL;
}

static bool real_matrix_ok(RealMatrix matrix) {
    return matrix.data != NULL;
}

static void set_identity(ComplexMatrix matrix) {
    int i;

    memset(matrix.data, 0, (size_t) matrix.n * (size_t) matrix.n * sizeof(complex double));
    for (i = 0; i < matrix.n; ++i) {
        matrix.data[IDX(i, i, matrix.n)] = 1.0;
    }
}

static void set_real_identity(RealMatrix matrix) {
    int i;

    memset(matrix.data, 0, (size_t) matrix.n * (size_t) matrix.n * sizeof(double));
    for (i = 0; i < matrix.n; ++i) {
        matrix.data[IDX(i, i, matrix.n)] = 1.0;
    }
}

static ComplexMatrix copy_from_pointer_rows(int n, complex double **rows) {
    ComplexMatrix matrix = make_complex_matrix(n);
    int i;
    int j;

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            matrix.data[IDX(i, j, n)] = rows[i][j];
        }
    }

    return matrix;
}

static ComplexMatrix build_extended_unitary(int l, complex double **base) {
    ComplexMatrix matrix = make_complex_matrix(2 * l);
    int i;
    int j;

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    for (i = 0; i < 2 * l; ++i) {
        matrix.data[IDX(i, i, 2 * l)] = 1.0;
    }

    for (i = 0; i < l; ++i) {
        for (j = 0; j < l; ++j) {
            matrix.data[IDX(i, j, 2 * l)] = base[i][j];
        }
    }

    return matrix;
}

static ComplexMatrix multiply_complex(ComplexMatrix left, ComplexMatrix right) {
    ComplexMatrix result = make_complex_matrix(left.n);
    int i;
    int j;
    int k;

    if (!matrix_ok(result)) {
        return result;
    }

    for (i = 0; i < left.n; ++i) {
        for (k = 0; k < left.n; ++k) {
            complex double left_value = left.data[IDX(i, k, left.n)];

            if (cabs(left_value) == 0.0) {
                continue;
            }

            for (j = 0; j < left.n; ++j) {
                result.data[IDX(i, j, left.n)] += left_value * right.data[IDX(k, j, left.n)];
            }
        }
    }

    return result;
}

static ComplexMatrix multiply_complex_right_conj_transpose(ComplexMatrix left, ComplexMatrix right) {
    ComplexMatrix result = make_complex_matrix(left.n);
    int i;
    int j;
    int k;

    if (!matrix_ok(result)) {
        return result;
    }

    for (i = 0; i < left.n; ++i) {
        for (j = 0; j < left.n; ++j) {
            complex double sum = 0.0;

            for (k = 0; k < left.n; ++k) {
                sum += left.data[IDX(i, k, left.n)] * conj(right.data[IDX(j, k, left.n)]);
            }

            result.data[IDX(i, j, left.n)] = sum;
        }
    }

    return result;
}

static ComplexMatrix multiply_complex_right_transpose(ComplexMatrix left, ComplexMatrix right) {
    ComplexMatrix result = make_complex_matrix(left.n);
    int i;
    int j;
    int k;

    if (!matrix_ok(result)) {
        return result;
    }

    for (i = 0; i < left.n; ++i) {
        for (j = 0; j < left.n; ++j) {
            complex double sum = 0.0;

            for (k = 0; k < left.n; ++k) {
                sum += left.data[IDX(i, k, left.n)] * right.data[IDX(j, k, left.n)];
            }

            result.data[IDX(i, j, left.n)] = sum;
        }
    }

    return result;
}

static ComplexMatrix multiply_complex_right_conjugate(ComplexMatrix left, ComplexMatrix right) {
    ComplexMatrix result = make_complex_matrix(left.n);
    int i;
    int j;
    int k;

    if (!matrix_ok(result)) {
        return result;
    }

    for (i = 0; i < left.n; ++i) {
        for (j = 0; j < left.n; ++j) {
            complex double sum = 0.0;

            for (k = 0; k < left.n; ++k) {
                sum += left.data[IDX(i, k, left.n)] * conj(right.data[IDX(k, j, left.n)]);
            }

            result.data[IDX(i, j, left.n)] = sum;
        }
    }

    return result;
}

static ComplexMatrix gate_single_squeeze_matrix(int n, int mode, double r, bool y_block) {
    ComplexMatrix matrix = make_complex_matrix(n);

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    if (!y_block) {
        set_identity(matrix);
        matrix.data[IDX(mode, mode, n)] = cosh(r);
    } else {
        matrix.data[IDX(mode, mode, n)] = -sinh(r);
    }

    return matrix;
}

static ComplexMatrix gate_beamsplitter_matrix(int n, int k, int l, double theta) {
    ComplexMatrix matrix = make_complex_matrix(n);
    double ch = cos(theta);
    double sh = sin(theta);

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    set_identity(matrix);
    matrix.data[IDX(k, k, n)] = ch;
    matrix.data[IDX(k, l, n)] = sh;
    matrix.data[IDX(l, k, n)] = -sh;
    matrix.data[IDX(l, l, n)] = ch;
    return matrix;
}

static bool compose_bogoliubov(
    ComplexMatrix x2,
    ComplexMatrix y2,
    ComplexMatrix *x1,
    ComplexMatrix *y1
) {
    ComplexMatrix x2x1 = multiply_complex(x2, *x1);
    ComplexMatrix y2y1c = multiply_complex_right_conjugate(y2, *y1);
    ComplexMatrix x2y1 = multiply_complex(x2, *y1);
    ComplexMatrix y2x1c = multiply_complex_right_conjugate(y2, *x1);
    ComplexMatrix next_x;
    ComplexMatrix next_y;
    int i;
    int entries;

    if (!matrix_ok(x2x1) || !matrix_ok(y2y1c) || !matrix_ok(x2y1) || !matrix_ok(y2x1c)) {
        free_complex_matrix(&x2x1);
        free_complex_matrix(&y2y1c);
        free_complex_matrix(&x2y1);
        free_complex_matrix(&y2x1c);
        return false;
    }

    next_x = make_complex_matrix(x1->n);
    next_y = make_complex_matrix(y1->n);
    if (!matrix_ok(next_x) || !matrix_ok(next_y)) {
        free_complex_matrix(&x2x1);
        free_complex_matrix(&y2y1c);
        free_complex_matrix(&x2y1);
        free_complex_matrix(&y2x1c);
        free_complex_matrix(&next_x);
        free_complex_matrix(&next_y);
        return false;
    }

    entries = x1->n * x1->n;
    for (i = 0; i < entries; ++i) {
        next_x.data[i] = x2x1.data[i] + y2y1c.data[i];
        next_y.data[i] = x2y1.data[i] + y2x1c.data[i];
    }

    free_complex_matrix(&x2x1);
    free_complex_matrix(&y2y1c);
    free_complex_matrix(&x2y1);
    free_complex_matrix(&y2x1c);

    free_complex_matrix(x1);
    free_complex_matrix(y1);
    *x1 = next_x;
    *y1 = next_y;
    return true;
}

static bool build_gaussian_mmat(int l, int *n, complex double **U, complex double **Up, double *ls, ComplexMatrix *mmat) {
    int nmodes = 2 * l;
    ComplexMatrix X = make_complex_matrix(nmodes);
    ComplexMatrix Y = make_complex_matrix(nmodes);
    ComplexMatrix Ue = build_extended_unitary(l, U);
    ComplexMatrix Uep = build_extended_unitary(l, Up);
    int i;

    if (!matrix_ok(X) || !matrix_ok(Y) || !matrix_ok(Ue) || !matrix_ok(Uep)) {
        free_complex_matrix(&X);
        free_complex_matrix(&Y);
        free_complex_matrix(&Ue);
        free_complex_matrix(&Uep);
        return false;
    }

    set_identity(X);

    for (i = 0; i < l; ++i) {
        double t = asinh(sqrt((double) n[i]));
        ComplexMatrix gate_x = gate_beamsplitter_matrix(nmodes, i, i + l, M_PI / 4.0);
        ComplexMatrix gate_y = make_complex_matrix(nmodes);
        if (!matrix_ok(gate_x) || !matrix_ok(gate_y) || !compose_bogoliubov(gate_x, gate_y, &X, &Y)) {
            free_complex_matrix(&gate_x);
            free_complex_matrix(&gate_y);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&gate_x);
        free_complex_matrix(&gate_y);

        gate_x = gate_single_squeeze_matrix(nmodes, i, t, false);
        gate_y = gate_single_squeeze_matrix(nmodes, i, t, true);
        if (!matrix_ok(gate_x) || !matrix_ok(gate_y) || !compose_bogoliubov(gate_x, gate_y, &X, &Y)) {
            free_complex_matrix(&gate_x);
            free_complex_matrix(&gate_y);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&gate_x);
        free_complex_matrix(&gate_y);

        gate_x = gate_single_squeeze_matrix(nmodes, i + l, -t, false);
        gate_y = gate_single_squeeze_matrix(nmodes, i + l, -t, true);
        if (!matrix_ok(gate_x) || !matrix_ok(gate_y) || !compose_bogoliubov(gate_x, gate_y, &X, &Y)) {
            free_complex_matrix(&gate_x);
            free_complex_matrix(&gate_y);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&gate_x);
        free_complex_matrix(&gate_y);

        gate_x = gate_beamsplitter_matrix(nmodes, i, i + l, -M_PI / 4.0);
        gate_y = make_complex_matrix(nmodes);
        if (!matrix_ok(gate_x) || !matrix_ok(gate_y) || !compose_bogoliubov(gate_x, gate_y, &X, &Y)) {
            free_complex_matrix(&gate_x);
            free_complex_matrix(&gate_y);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&gate_x);
        free_complex_matrix(&gate_y);
    }

    {
        ComplexMatrix zero = make_complex_matrix(nmodes);
        if (!matrix_ok(zero) || !compose_bogoliubov(Uep, zero, &X, &Y)) {
            free_complex_matrix(&zero);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&zero);
    }

    for (i = 0; i < l; ++i) {
        ComplexMatrix gate_x = gate_single_squeeze_matrix(nmodes, i, -ls[i], false);
        ComplexMatrix gate_y = gate_single_squeeze_matrix(nmodes, i, -ls[i], true);
        if (!matrix_ok(gate_x) || !matrix_ok(gate_y) || !compose_bogoliubov(gate_x, gate_y, &X, &Y)) {
            free_complex_matrix(&gate_x);
            free_complex_matrix(&gate_y);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&gate_x);
        free_complex_matrix(&gate_y);
    }

    {
        ComplexMatrix zero = make_complex_matrix(nmodes);
        ComplexMatrix result;
        if (!matrix_ok(zero) || !compose_bogoliubov(Ue, zero, &X, &Y)) {
            free_complex_matrix(&zero);
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        free_complex_matrix(&zero);
        result = multiply_complex_right_transpose(X, Y);
        if (!matrix_ok(result)) {
            free_complex_matrix(&X);
            free_complex_matrix(&Y);
            free_complex_matrix(&Ue);
            free_complex_matrix(&Uep);
            return false;
        }
        memcpy(mmat->data, result.data, (size_t) nmodes * (size_t) nmodes * sizeof(complex double));
        free_complex_matrix(&result);
    }

    free_complex_matrix(&X);
    free_complex_matrix(&Y);
    free_complex_matrix(&Ue);
    free_complex_matrix(&Uep);
    return true;
}

static void apply_u(ComplexMatrix *nmat, ComplexMatrix *mmat, ComplexMatrix u) {
    ComplexMatrix tmp_n = multiply_complex(u, *nmat);
    ComplexMatrix next_n;
    ComplexMatrix tmp_m = multiply_complex(u, *mmat);
    ComplexMatrix next_m;

    if (!matrix_ok(tmp_n) || !matrix_ok(tmp_m)) {
        free_complex_matrix(&tmp_n);
        free_complex_matrix(&tmp_m);
        return;
    }

    next_n = multiply_complex_right_conj_transpose(tmp_n, u);
    next_m = multiply_complex_right_transpose(tmp_m, u);

    free_complex_matrix(&tmp_n);
    free_complex_matrix(&tmp_m);

    if (!matrix_ok(next_n) || !matrix_ok(next_m)) {
        free_complex_matrix(&next_n);
        free_complex_matrix(&next_m);
        return;
    }

    memcpy(nmat->data, next_n.data, (size_t) nmat->n * (size_t) nmat->n * sizeof(complex double));
    memcpy(mmat->data, next_m.data, (size_t) mmat->n * (size_t) mmat->n * sizeof(complex double));

    free_complex_matrix(&next_n);
    free_complex_matrix(&next_m);
}

static void squeeze_mode(ComplexMatrix *nmat, ComplexMatrix *mmat, int k, double r) {
    int l;
    int n = nmat->n;
    complex double phase = 1.0;
    complex double phase_squared = 1.0;
    double sh = sinh(r);
    double ch = cosh(r);
    double sh_squared = sh * sh;
    double ch_squared = ch * ch;
    double sh_ch = sh * ch;
    complex double *nk = malloc((size_t) n * sizeof(complex double));
    complex double *mk = malloc((size_t) n * sizeof(complex double));

    if (nk == NULL || mk == NULL) {
        free(nk);
        free(mk);
        return;
    }

    for (l = 0; l < n; ++l) {
        nk[l] = nmat->data[IDX(k, l, n)];
        mk[l] = mmat->data[IDX(k, l, n)];
    }

    nmat->data[IDX(k, k, n)] =
        sh_squared
        - phase * sh_ch * conj(mk[k])
        - sh_ch * conj(phase) * mk[k]
        + ch_squared * nk[k]
        + sh_squared * nk[k];
    mmat->data[IDX(k, k, n)] =
        -(phase * sh_ch)
        + phase_squared * sh_squared * conj(mk[k])
        + ch_squared * mk[k]
        - 2.0 * phase * sh_ch * nk[k];

    for (l = 0; l < n; ++l) {
        if (l == k) {
            continue;
        }

        nmat->data[IDX(k, l, n)] = -(sh * conj(phase) * mk[l]) + ch * nk[l];
        mmat->data[IDX(k, l, n)] = ch * mk[l] - phase * sh * nk[l];
    }

    for (l = 0; l < n; ++l) {
        nmat->data[IDX(l, k, n)] = conj(nmat->data[IDX(k, l, n)]);
        mmat->data[IDX(l, k, n)] = mmat->data[IDX(k, l, n)];
    }

    free(nk);
    free(mk);
}

static void beamsplitter_modes(ComplexMatrix *nmat, ComplexMatrix *mmat, int k, int l, double theta) {
    int i;
    int n = nmat->n;
    complex double phase = 1.0;
    complex double phase_squared = 1.0;
    double sh = sin(theta);
    double ch = cos(theta);
    double sh_squared = sh * sh;
    double ch_squared = ch * ch;
    double sh_ch = sh * ch;
    complex double *nk = malloc((size_t) n * sizeof(complex double));
    complex double *mk = malloc((size_t) n * sizeof(complex double));
    complex double *nl = malloc((size_t) n * sizeof(complex double));
    complex double *ml = malloc((size_t) n * sizeof(complex double));

    if (nk == NULL || mk == NULL || nl == NULL || ml == NULL) {
        free(nk);
        free(mk);
        free(nl);
        free(ml);
        return;
    }

    for (i = 0; i < n; ++i) {
        nk[i] = nmat->data[IDX(k, i, n)];
        mk[i] = mmat->data[IDX(k, i, n)];
        nl[i] = nmat->data[IDX(l, i, n)];
        ml[i] = mmat->data[IDX(l, i, n)];
    }

    nmat->data[IDX(k, k, n)] =
        ch_squared * nk[k] + phase * sh_ch * nk[l] + sh_ch * conj(phase) * nl[k] + sh_squared * nl[l];
    nmat->data[IDX(k, l, n)] =
        -(sh_ch * conj(phase) * nk[k])
        + ch_squared * nk[l]
        - sh_squared * conj(phase_squared) * nl[k]
        + sh_ch * conj(phase) * nl[l];
    nmat->data[IDX(l, k, n)] = conj(nmat->data[IDX(k, l, n)]);
    nmat->data[IDX(l, l, n)] =
        sh_squared * nk[k] - phase * sh_ch * nk[l] - sh_ch * conj(phase) * nl[k] + ch_squared * nl[l];

    mmat->data[IDX(k, k, n)] = ch_squared * mk[k] + 2.0 * phase * sh_ch * ml[k] + phase_squared * sh_squared * ml[l];
    mmat->data[IDX(k, l, n)] =
        -(sh_ch * conj(phase) * mk[k]) + ch_squared * ml[k] - sh_squared * ml[k] + phase * sh_ch * ml[l];
    mmat->data[IDX(l, k, n)] = mmat->data[IDX(k, l, n)];
    mmat->data[IDX(l, l, n)] =
        sh_squared * conj(phase_squared) * mk[k] - 2.0 * sh_ch * conj(phase) * ml[k] + ch_squared * ml[l];

    for (i = 0; i < n; ++i) {
        if (i == k || i == l) {
            continue;
        }

        nmat->data[IDX(k, i, n)] = ch * nk[i] + sh * conj(phase) * nl[i];
        mmat->data[IDX(k, i, n)] = ch * mk[i] + phase * sh * ml[i];
        nmat->data[IDX(l, i, n)] = -(phase * sh * nk[i]) + ch * nl[i];
        mmat->data[IDX(l, i, n)] = -(sh * conj(phase) * mk[i]) + ch * ml[i];
    }

    for (i = 0; i < n; ++i) {
        nmat->data[IDX(i, k, n)] = conj(nmat->data[IDX(k, i, n)]);
        mmat->data[IDX(i, k, n)] = mmat->data[IDX(k, i, n)];
        nmat->data[IDX(i, l, n)] = conj(nmat->data[IDX(l, i, n)]);
        mmat->data[IDX(i, l, n)] = mmat->data[IDX(l, i, n)];
    }

    free(nk);
    free(mk);
    free(nl);
    free(ml);
}

static void apply_tmsq(ComplexMatrix *nmat, ComplexMatrix *mmat, int i, int j, double r) {
    beamsplitter_modes(nmat, mmat, i, j, M_PI / 4.0);
    squeeze_mode(nmat, mmat, i, -r);
    squeeze_mode(nmat, mmat, j, r);
    beamsplitter_modes(nmat, mmat, i, j, -M_PI / 4.0);
}

static void swap_real_columns(RealMatrix *vectors, int col_a, int col_b) {
    int row;
    int n = vectors->n;

    for (row = 0; row < n; ++row) {
        double tmp = vectors->data[IDX(row, col_a, n)];
        vectors->data[IDX(row, col_a, n)] = vectors->data[IDX(row, col_b, n)];
        vectors->data[IDX(row, col_b, n)] = tmp;
    }
}

static void jacobi_eigendecompose(RealMatrix *matrix, double *eigenvalues, RealMatrix *eigenvectors) {
    int n = matrix->n;
    int sweep;
    const int max_sweeps = 100 * n * n;
    const double tol = 1e-12;

    set_real_identity(*eigenvectors);

    for (sweep = 0; sweep < max_sweeps; ++sweep) {
        int p;
        int q;
        double max_offdiag = 0.0;

        for (p = 0; p < n; ++p) {
            for (q = p + 1; q < n; ++q) {
                double value = fabs(matrix->data[IDX(p, q, n)]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                }
            }
        }

        if (max_offdiag < tol) {
            break;
        }

        for (p = 0; p < n - 1; ++p) {
            for (q = p + 1; q < n; ++q) {
                double apq = matrix->data[IDX(p, q, n)];

                if (fabs(apq) < tol) {
                    continue;
                }

                {
                    double app = matrix->data[IDX(p, p, n)];
                    double aqq = matrix->data[IDX(q, q, n)];
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau >= 0.0)
                        ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                        : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                    double c = 1.0 / sqrt(1.0 + t * t);
                    double s = t * c;
                    int r;

                    for (r = 0; r < n; ++r) {
                        if (r == p || r == q) {
                            continue;
                        }

                        {
                            double arp = matrix->data[IDX(r, p, n)];
                            double arq = matrix->data[IDX(r, q, n)];
                            double new_rp = c * arp - s * arq;
                            double new_rq = s * arp + c * arq;

                            matrix->data[IDX(r, p, n)] = new_rp;
                            matrix->data[IDX(p, r, n)] = new_rp;
                            matrix->data[IDX(r, q, n)] = new_rq;
                            matrix->data[IDX(q, r, n)] = new_rq;
                        }
                    }

                    matrix->data[IDX(p, p, n)] = app - t * apq;
                    matrix->data[IDX(q, q, n)] = aqq + t * apq;
                    matrix->data[IDX(p, q, n)] = 0.0;
                    matrix->data[IDX(q, p, n)] = 0.0;

                    for (r = 0; r < n; ++r) {
                        double vrp = eigenvectors->data[IDX(r, p, n)];
                        double vrq = eigenvectors->data[IDX(r, q, n)];
                        eigenvectors->data[IDX(r, p, n)] = c * vrp - s * vrq;
                        eigenvectors->data[IDX(r, q, n)] = s * vrp + c * vrq;
                    }
                }
            }
        }
    }

    for (sweep = 0; sweep < n; ++sweep) {
        eigenvalues[sweep] = matrix->data[IDX(sweep, sweep, n)];
    }

    for (sweep = 0; sweep < n - 1; ++sweep) {
        int best = sweep;
        int idx;

        for (idx = sweep + 1; idx < n; ++idx) {
            if (eigenvalues[idx] > eigenvalues[best]) {
                best = idx;
            }
        }

        if (best != sweep) {
            double tmp = eigenvalues[sweep];
            eigenvalues[sweep] = eigenvalues[best];
            eigenvalues[best] = tmp;
            swap_real_columns(eigenvectors, sweep, best);
        }
    }
}

static bool takagi_from_symmetric(ComplexMatrix matrix, double *sigmas, ComplexMatrix *unitary) {
    int n = matrix.n;
    int doubled = 2 * n;
    RealMatrix block = make_real_matrix(doubled);
    RealMatrix eigenvectors = make_real_matrix(doubled);
    double *eigenvalues = calloc((size_t) doubled, sizeof(double));
    complex double *columns = calloc((size_t) n * (size_t) n, sizeof(complex double));
    int i;
    int j;
    int col_index = 0;
    const double tol = 1e-10;

    if (!real_matrix_ok(block) || !real_matrix_ok(eigenvectors) || eigenvalues == NULL || columns == NULL) {
        free_real_matrix(&block);
        free_real_matrix(&eigenvectors);
        free(eigenvalues);
        free(columns);
        return false;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            complex double value = matrix.data[IDX(i, j, n)];
            block.data[IDX(i, j, doubled)] = creal(value);
            block.data[IDX(i, j + n, doubled)] = cimag(value);
            block.data[IDX(i + n, j, doubled)] = cimag(value);
            block.data[IDX(i + n, j + n, doubled)] = -creal(value);
        }
    }

    jacobi_eigendecompose(&block, eigenvalues, &eigenvectors);

    for (j = 0; j < doubled && col_index < n; ++j) {
        complex double *column = columns + (size_t) col_index * (size_t) n;
        double norm_sq = 0.0;
        int previous;

        if (eigenvalues[j] < -tol) {
            continue;
        }

        for (i = 0; i < n; ++i) {
            column[i] = eigenvectors.data[IDX(i, j, doubled)] + I * eigenvectors.data[IDX(i + n, j, doubled)];
            norm_sq += pow(cabs(column[i]), 2.0);
        }

        if (norm_sq < tol) {
            continue;
        }

        for (previous = 0; previous < col_index; ++previous) {
            complex double overlap = 0.0;

            for (i = 0; i < n; ++i) {
                overlap += conj(columns[(size_t) previous * (size_t) n + (size_t) i]) * column[i];
            }

            for (i = 0; i < n; ++i) {
                column[i] -= overlap * columns[(size_t) previous * (size_t) n + (size_t) i];
            }
        }

        norm_sq = 0.0;
        for (i = 0; i < n; ++i) {
            norm_sq += pow(cabs(column[i]), 2.0);
        }

        if (norm_sq < tol) {
            continue;
        }

        {
            double norm = sqrt(norm_sq);
            for (i = 0; i < n; ++i) {
                column[i] /= norm;
            }
        }

        sigmas[col_index] = eigenvalues[j] > 0.0 ? eigenvalues[j] : 0.0;
        ++col_index;
    }

    while (col_index < n) {
        int basis = col_index;
        complex double *column = columns + (size_t) col_index * (size_t) n;
        int previous;
        double norm_sq = 0.0;

        for (i = 0; i < n; ++i) {
            column[i] = (i == basis) ? 1.0 : 0.0;
        }

        for (previous = 0; previous < col_index; ++previous) {
            complex double overlap = 0.0;

            for (i = 0; i < n; ++i) {
                overlap += conj(columns[(size_t) previous * (size_t) n + (size_t) i]) * column[i];
            }

            for (i = 0; i < n; ++i) {
                column[i] -= overlap * columns[(size_t) previous * (size_t) n + (size_t) i];
            }
        }

        for (i = 0; i < n; ++i) {
            norm_sq += pow(cabs(column[i]), 2.0);
        }

        if (norm_sq < tol) {
            free_real_matrix(&block);
            free_real_matrix(&eigenvectors);
            free(eigenvalues);
            free(columns);
            return false;
        }

        {
            double norm = sqrt(norm_sq);
            for (i = 0; i < n; ++i) {
                column[i] /= norm;
            }
        }

        sigmas[col_index] = 0.0;
        ++col_index;
    }

    for (j = 0; j < n; ++j) {
        for (i = 0; i < n; ++i) {
            unitary->data[IDX(i, j, n)] = columns[(size_t) j * (size_t) n + (size_t) i];
        }
    }

    free_real_matrix(&block);
    free_real_matrix(&eigenvectors);
    free(eigenvalues);
    free(columns);
    return true;
}

static ComplexMatrix build_bargmann_matrix(ComplexMatrix takagi_u, const double *sigmas, double *log_cosh_product) {
    ComplexMatrix matrix = make_complex_matrix(takagi_u.n);
    int i;
    int j;
    int k;

    *log_cosh_product = 0.0;

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    for (k = 0; k < takagi_u.n; ++k) {
        double lt = -0.5 * asinh(2.0 * sigmas[k]);
        double coeff = tanh(lt);

        *log_cosh_product += log(cosh(lt));

        if (fabs(coeff) < 1e-16) {
            continue;
        }

        for (i = 0; i < takagi_u.n; ++i) {
            complex double left = conj(takagi_u.data[IDX(i, k, takagi_u.n)]);

            for (j = 0; j < takagi_u.n; ++j) {
                complex double right = conj(takagi_u.data[IDX(j, k, takagi_u.n)]);
                matrix.data[IDX(i, j, takagi_u.n)] += coeff * left * right;
            }
        }
    }

    return matrix;
}

static complex double *apply_bargmann_shift(ComplexMatrix bmat, const complex double *alphat) {
    complex double *zeta = calloc((size_t) bmat.n, sizeof(complex double));
    int i;
    int j;

    if (zeta == NULL) {
        return NULL;
    }

    for (i = 0; i < bmat.n; ++i) {
        complex double value = alphat[i];

        for (j = 0; j < bmat.n; ++j) {
            value -= bmat.data[IDX(i, j, bmat.n)] * conj(alphat[j]);
        }

        zeta[i] = value;
    }

    return zeta;
}

static int *build_sp_list(int total_modes, const int *p, int *sp_size) {
    int total = 0;
    int k;
    int cursor = 0;
    int *sp;

    for (k = 0; k < total_modes; ++k) {
        if (p[k] < 0) {
            return NULL;
        }
        total += p[k];
    }

    *sp_size = total;
    sp = calloc((size_t) total, sizeof(int));
    if (sp == NULL && total > 0) {
        return NULL;
    }

    for (k = 0; k < total_modes; ++k) {
        int count;
        for (count = 0; count < p[k]; ++count) {
            sp[cursor++] = k;
        }
    }

    return sp;
}

static ComplexMatrix build_repeated_matrix(ComplexMatrix bmat, const complex double *zeta, const int *sp, int sp_size) {
    ComplexMatrix matrix = make_complex_matrix(sp_size);
    int i;
    int j;

    if (!matrix_ok(matrix)) {
        return matrix;
    }

    for (i = 0; i < sp_size; ++i) {
        for (j = 0; j < sp_size; ++j) {
            matrix.data[IDX(i, j, sp_size)] = bmat.data[IDX(sp[i], sp[j], bmat.n)];
        }
        matrix.data[IDX(i, i, sp_size)] = zeta[sp[i]];
    }

    return matrix;
}

static complex double loop_hafnian_recursive(const complex double *matrix, int n, uint64_t mask, complex double *memo, unsigned char *seen) {
    int i;
    uint64_t rest;
    uint64_t partners;
    complex double result;

    if (mask == 0) {
        return 1.0;
    }

    if (seen[mask]) {
        return memo[mask];
    }

    i = __builtin_ctzll(mask);
    rest = mask & ~(1ULL << i);
    result = matrix[IDX(i, i, n)] * loop_hafnian_recursive(matrix, n, rest, memo, seen);

    partners = rest;
    while (partners != 0) {
        int j = __builtin_ctzll(partners);
        uint64_t paired = rest & ~(1ULL << j);
        result += matrix[IDX(i, j, n)] * loop_hafnian_recursive(matrix, n, paired, memo, seen);
        partners &= partners - 1;
    }

    seen[mask] = 1;
    memo[mask] = result;
    return result;
}

static complex double loop_hafnian_slow(const complex double *matrix, const int *indices, int count, int stride) {
    int pos;
    complex double result;

    if (count == 0) {
        return 1.0;
    }

    result = matrix[IDX(indices[0], indices[0], stride)] * loop_hafnian_slow(matrix, indices + 1, count - 1, stride);

    for (pos = 1; pos < count; ++pos) {
        int *reduced = calloc((size_t) (count - 2), sizeof(int));
        int cursor = 0;
        int idx;
        complex double contribution;

        if (reduced == NULL && count > 2) {
            return CMPLX(NAN, NAN);
        }

        for (idx = 1; idx < count; ++idx) {
            if (idx == pos) {
                continue;
            }
            reduced[cursor++] = indices[idx];
        }

        contribution = matrix[IDX(indices[0], indices[pos], stride)] * loop_hafnian_slow(matrix, reduced, count - 2, stride);
        result += contribution;
        free(reduced);
    }

    return result;
}

static complex double loop_hafnian(ComplexMatrix matrix) {
    if (matrix.n == 0) {
        return 1.0;
    }

    if (matrix.n <= 24) {
        size_t table_size = (size_t) 1ULL << matrix.n;
        complex double *memo = calloc(table_size, sizeof(complex double));
        unsigned char *seen = calloc(table_size, sizeof(unsigned char));
        complex double value;

        if (memo == NULL || seen == NULL) {
            free(memo);
            free(seen);
            return CMPLX(NAN, NAN);
        }

        value = loop_hafnian_recursive(matrix.data, matrix.n, (1ULL << matrix.n) - 1ULL, memo, seen);
        free(memo);
        free(seen);
        return value;
    }

    {
        int *indices = calloc((size_t) matrix.n, sizeof(int));
        complex double value;
        int i;

        if (indices == NULL) {
            return CMPLX(NAN, NAN);
        }

        for (i = 0; i < matrix.n; ++i) {
            indices[i] = i;
        }

        value = loop_hafnian_slow(matrix.data, indices, matrix.n, matrix.n);
        free(indices);
        return value;
    }
}

complex double matelem(
    int l,
    int *m,
    int *n,
    complex double **U,
    complex double **p,
    double *ls,
    complex double *alpha
) {
    int nmodes;
    ComplexMatrix mmat;
    double *ts;
    double *sigmas;
    ComplexMatrix takagi_u;
    ComplexMatrix bmat;
    complex double *alphat;
    complex double *zeta;
    int *pn;
    int *sp;
    int sp_size = 0;
    ComplexMatrix bp;
    double log_r = 0.0;
    double log_prefns = 0.0;
    double log_cosh_product = 0.0;
    complex double pref = 0.0;
    complex double amp;
    complex double result;
    int i;

    if (l <= 0 || m == NULL || n == NULL || U == NULL || p == NULL || ls == NULL || alpha == NULL) {
        return invalid_result();
    }

    nmodes = 2 * l;
    mmat = make_complex_matrix(nmodes);
    ts = calloc((size_t) l, sizeof(double));
    sigmas = calloc((size_t) nmodes, sizeof(double));
    takagi_u = make_complex_matrix(nmodes);
    alphat = calloc((size_t) nmodes, sizeof(complex double));
    pn = calloc((size_t) nmodes, sizeof(int));
    bp.n = 0;
    bp.data = NULL;

    if (!matrix_ok(mmat) || ts == NULL || sigmas == NULL || !matrix_ok(takagi_u) || alphat == NULL || pn == NULL) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free(ts);
        free(sigmas);
        free(alphat);
        free(pn);
        return invalid_result();
    }

    for (i = 0; i < l; ++i) {
        double ni;

        if (m[i] < 0 || n[i] < 0) {
            free_complex_matrix(&mmat);
            free_complex_matrix(&takagi_u);
            free(ts);
            free(sigmas);
            free(alphat);
            free(pn);
            return invalid_result();
        }

        ni = (double) n[i];
        ts[i] = asinh(sqrt(ni));
        pn[i] = m[i];
        pn[i + l] = n[i];
        alphat[i] = alpha[i];

        if (n[i] > 0) {
            log_r -= 0.5 * ni * (log(ni) - log(ni + 1.0)) - 0.5 * log(ni + 1.0);
        }

        log_prefns += 0.5 * lgamma((double) pn[i] + 1.0);
        log_prefns += 0.5 * lgamma((double) pn[i + l] + 1.0);
    }

    if (!build_gaussian_mmat(l, n, U, p, ls, &mmat)) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free(ts);
        free(sigmas);
        free(alphat);
        free(pn);
        return invalid_result();
    }

    if (!takagi_from_symmetric(mmat, sigmas, &takagi_u)) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free(ts);
        free(sigmas);
        free(alphat);
        free(pn);
        return invalid_result();
    }

    bmat = build_bargmann_matrix(takagi_u, sigmas, &log_cosh_product);
    if (!matrix_ok(bmat)) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free(ts);
        free(sigmas);
        free(alphat);
        free(pn);
        return invalid_result();
    }

    zeta = apply_bargmann_shift(bmat, alphat);
    if (zeta == NULL) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free_complex_matrix(&bmat);
        free(ts);
        free(sigmas);
        free(alphat);
        free(pn);
        return invalid_result();
    }

    for (i = 0; i < nmodes; ++i) {
        pref += conj(alphat[i]) * zeta[i];
    }
    pref *= -0.5;

    sp = build_sp_list(nmodes, pn, &sp_size);
    if (sp == NULL && sp_size > 0) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free_complex_matrix(&bmat);
        free(ts);
        free(sigmas);
        free(alphat);
        free(zeta);
        free(pn);
        return invalid_result();
    }

    bp = build_repeated_matrix(bmat, zeta, sp, sp_size);
    if (sp_size > 0 && !matrix_ok(bp)) {
        free_complex_matrix(&mmat);
        free_complex_matrix(&takagi_u);
        free_complex_matrix(&bmat);
        free(ts);
        free(sigmas);
        free(alphat);
        free(zeta);
        free(pn);
        free(sp);
        return invalid_result();
    }

    amp = loop_hafnian(bp);
    result = cexp(pref + log_r - log_prefns - 0.5 * log_cosh_product) * amp;

    free_complex_matrix(&mmat);
    free_complex_matrix(&takagi_u);
    free_complex_matrix(&bmat);
    free_complex_matrix(&bp);
    free(ts);
    free(sigmas);
    free(alphat);
    free(zeta);
    free(pn);
    free(sp);

    return result;
}
