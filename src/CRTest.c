#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <Rmath.h>
#include <stdio.h>
#include <stdlib.h>

void _Gram(double *x, int *param, double *xbyx, double **xTx)
{
    // input: 
    // x in R^{n*p}, saved by row.
    // param = c(n, p).
    // xbyx is the upper triangle of x^T x without diagonal, saved by row.
    // xTx is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    unsigned int n, p, i, j, k, c;
    n = param[0];
    p = param[1];
    c = 1;
    double tmp, *xi, *xj;

    for (i = 0; i < n-1; ++i) {
        xi = x + i*p; // point to i-th sample
        for (j = i+1; j < n; ++j) {
            xj = x + j*p; // point to j-th sample
            for (tmp = 0.0, k = 0; k < p; ++k) 
                tmp += xi[k]*xj[k];
            xbyx[c] = tmp;
            c++;
        }
    }

    c = 0;
    for (i = 0; i < n-1; ++i) {
        xTx[i] = xbyx + c;
        c += n-2-i;
    }
}


double sigmoid_der(double z) {
    double val = 0.0;
    if (z > 0) {
        val = exp(-z);
    } else {
        val = exp(z);
    }
    val = 1.0 / (val + 1.0/val + 2.0);
    return(val);
}

void _Rank_part_sigmoid(int *S, double *Y, double *z, int n, double *rij, double **R, double *ri)
{
    // input: 
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    // rij in R^{n(n-1)/2} is the upper triangle of Rij without diagonal, by row.
    // Rij is the matrix of partial rank R_{i,j}, 
    // only saves the upper triangle without diagonal, saved by row.
    // ri = sum_{j \ne i} R_{i, j}
    unsigned int i, j, c;
    c = 1;
    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            if ((S[i] == 0) || (Y[i] == Y[j])) {
                rij[c] = 0;
            } else {
                rij[c] = -sigmoid_der(z[j] - z[i]);
            }
            c++;
        }
    }

    c = 0;
    for (i = 0; i < n-1; ++i) {
        R[i] = rij + c;
        c += n-2-i;
    }

    // remember that R[i][j] + R[j][i] = 0
    for (i = 0; i < n; ++i) {
        ri[i] = 0;
        for (j = 0; j < i; ++j)   ri[i] -= R[j][i];
        for (j = i+1; j < n; ++j) ri[i] += R[i][j];
    }
}

double norm_der(double z) {
    return(exp(-0.5 * z * z));
}

void _Rank_part_norm(int *S, double *Y, double *z, int n, double *rij, double **R, double *ri)
{
    // input: 
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    // rij in R^{n(n-1)/2} is the upper triangle of Rij without diagonal, by row.
    // Rij is the matrix of partial rank R_{i,j}, 
    // only saves the upper triangle without diagonal, saved by row.
    // ri = sum_{j \ne i} R_{i, j}
    unsigned int i, j, c;
    c = 1;
    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            if ((S[i] == 0) || (Y[i] == Y[j])) {
                rij[c] = 0;
            } else {
                rij[c] = -norm_der(z[j] - z[i]);
            }
            c++;
        }
    }

    c = 0;
    for (i = 0; i < n-1; ++i) {
        R[i] = rij + c;
        c += n-2-i;
    }

    // remember that R[i][j] + R[j][i] = 0
    for (i = 0; i < n; ++i) {
        ri[i] = 0;
        for (j = 0; j < i; ++j)   ri[i] -= R[j][i];
        for (j = i+1; j < n; ++j) ri[i] += R[i][j];
    }
}

void _Rank_1(int *S, double *Y, int n, int *rij, int **R, int *ri)
{
    // input: 
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    // rij in R^{n(n-1)/2} is the upper triangle of Rij without diagonal, by row.
    // Rij is the matrix of partial rank R_{i,j}, 
    // only saves the upper triangle without diagonal, saved by row.
    // ri = sum_{j \ne i} R_{i, j}
    unsigned int i, j, c;
    c = 1;
    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            if ((S[i] == 0) || (Y[i] == Y[j])) {
                rij[c] = 0;
            } else {
                rij[c] = -1;
            }
            c++;
        }
    }
    c = 0;
    for (i = 0; i < n-1; ++i) {
        R[i] = rij + c;
        c += n-2-i;
    }
    // remember that R[i][j] + R[j][i] = 0
    for (i = 0; i < n; ++i) {
        ri[i] = 0;
        for (j = 0; j < i; ++j)   ri[i] -= R[j][i];
        for (j = i+1; j < n; ++j) ri[i] += R[i][j];
    }
}

void _Rank_2(double *Y, int n, int *rij, int **R, int *ri)
{
    // input: 
    // Y has ascending ordered, Y_i <= Y_j
    // rij in R^{n(n-1)/2} is the upper triangle of Rij without diagonal, by row.
    // Rij is the matrix of R_{i,j} = I(Y[i] >= Y[j]) - I(Y[j] >= Y[i])
    // only saves the upper triangle without diagonal, saved by row.
    // ri = sum_{j \ne i} R_{i, j}
    unsigned int i, j, c;
    c = 1;
    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            if (Y[i] != Y[j]) {
                rij[c] = -1;
            } else {
                rij[c] = 0;
            }
            c++;
        }
    }

    c = 0;
    for (i = 0; i < n-1; ++i) {
        R[i] = rij + c;
        c += n-2-i;
    }

    // remember that R[i][j] + R[j][i] = 0
    for (i = 0; i < n; ++i) {
        ri[i] = 0;
        for (j = 0; j < i; ++j)   ri[i] -= R[j][i];
        for (j = i+1; j < n; ++j) ri[i] += R[i][j];
    }
}


double _spr_test(double **xTx, int *S, double *Y, int n, int *ranktype, 
    double *sigmaR)
{   
    // input: 
    // xTx in R^{n*n}, is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    int   *ri;
    int   *rank;
    int   **R;
    rank    = (int*)malloc(sizeof(int) * (n*n-n+2)/2);
    ri     = (int*)malloc(sizeof(int) *n);
    R      = (int**)malloc(sizeof(int*)*(n-1));
    if (ranktype[0] == 1) {
        _Rank_1(S, Y, n, rank, R, ri);
    } else if (ranktype[0] == 2) {
        _Rank_2(Y, n, rank, R, ri);
    }

    // printarray(ri, n);

    unsigned int i, j, k;
    int ell;
    double Test = 0.0;
    unsigned int tmp = (n-2)*(n-3);

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            ell = (ri[i] - R[i][j])*(ri[j] + R[i][j]);
            for (k = 0; k < i; ++k)   ell -= R[k][i] * R[k][j];
            for (k = i+1; k < j; ++k) ell += R[i][k] * R[k][j];
            for (k = j+1; k < n; ++k) ell -= R[i][k] * R[j][k];
            Test += ell * xTx[i][j] / tmp;
        }
    }

    double temp;
    sigmaR[0] = 0.0;
    for (i = 0; i < n; ++i) {
        temp = ri[i] / (n-1.0);
        sigmaR[0] += temp*temp;
    }
    sigmaR[0] /= n*1.0;

    Test /= n*(n-1)/2.0;

    free(ri);
    free(rank);
    free(R);
    return Test;
}

double _spr_test_part(double **xTx, int *S, double *Y, int n, int *ranktype, 
    double *sigmaR, double *z)
{   
    // input: 
    // xTx in R^{n*n}, is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    double   *ri;
    double   *rank;
    double   **R;
    rank = (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    ri   = (double*)malloc(sizeof(double) *n);
    R    = (double**)malloc(sizeof(double*)*(n-1));
    if (ranktype[0] == 1) {
        _Rank_part_sigmoid(S, Y, z, n, rank, R, ri);
    } else if (ranktype[0] == 2) {
        _Rank_part_norm(S, Y, z, n, rank, R, ri);
    }

    // printarray(ri, n);

    unsigned int i, j, k;
    double ell;
    double Test = 0.0;
    unsigned int tmp = (n-2)*(n-3);

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            ell = (ri[i] - R[i][j])*(ri[j] + R[i][j]);
            for (k = 0; k < i; ++k)   ell -= R[k][i] * R[k][j];
            for (k = i+1; k < j; ++k) ell += R[i][k] * R[k][j];
            for (k = j+1; k < n; ++k) ell -= R[i][k] * R[j][k];
            Test += ell * xTx[i][j] / tmp;
        }
    }

    double temp;
    sigmaR[0] = 0.0;
    for (i = 0; i < n; ++i) {
        temp = ri[i] / (n-1.0);
        sigmaR[0] += temp*temp;
    }
    sigmaR[0] /= n*1.0;

    Test /= n*(n-1)/2.0;

    free(ri);
    free(rank);
    free(R);
    return Test;
}

double _tr_sigma2_hat(double **xTx, int n)
{
    int i, j, k, l;
    double trsigma2, tmp, y1n, y2n, y3n;
    y1n = y2n = y3n = 0.0;

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            y1n += xTx[i][j] * xTx[i][j];
            for (k = j+1; k < n; ++k) {
                tmp = (xTx[i][j] + xTx[i][k])*xTx[j][k] + xTx[i][j]*xTx[i][k];
                y2n += tmp/(n-2);
                for (l = k+1; l < n; ++l) {
                    tmp = xTx[i][j]*xTx[k][l] + xTx[i][k]*xTx[j][l] + xTx[i][l]*xTx[j][k];
                    y3n += tmp*4/(n-2)/(n-3);
                }
            }
        }
    }

    trsigma2 = (y1n - y2n*2 + y3n)/(n*(n-1)/2);
    return trsigma2;
}


SEXP _SPR_Test(SEXP _X, SEXP _Y, SEXP _S, SEXP _Param, SEXP _Type)
{
    int n;
    n = INTEGER(_Param)[0];
    double *y;
    y = REAL(_Y);
    int *s, *type;
    s = INTEGER(_S);
    type = INTEGER(_Type);

    double *xbyx;
    double **xTx;
    xbyx     =  (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    xTx      = (double**)malloc(sizeof(double*)*(n-1));

    SEXP _Test, _Tr_sigma2, _sigmaR;
    PROTECT(_Test = allocVector(REALSXP, 1));
    PROTECT(_Tr_sigma2 = allocVector(REALSXP, 1));
    PROTECT(_sigmaR = allocVector(REALSXP, 1));

    _Gram(REAL(_X), INTEGER(_Param), xbyx, xTx);
    REAL(_Test)[0] = _spr_test(xTx, s, y, n, type, REAL(_sigmaR));
    REAL(_Tr_sigma2)[0] = _tr_sigma2_hat(xTx, n);

    SEXP Result, R_names;
    char *names[3] = {"test", "tr_sigma", "sigmaR"};
    PROTECT(Result    = allocVector(VECSXP,  3));
    PROTECT(R_names   = allocVector(STRSXP,  3));
    SET_STRING_ELT(R_names, 0,  mkChar(names[0]));
    SET_STRING_ELT(R_names, 1,  mkChar(names[1]));
    SET_STRING_ELT(R_names, 2,  mkChar(names[2]));
    SET_VECTOR_ELT(Result, 0, _Test);
    SET_VECTOR_ELT(Result, 1, _Tr_sigma2);
    SET_VECTOR_ELT(Result, 2, _sigmaR);
    setAttrib(Result, R_NamesSymbol, R_names); 

    free(xbyx);
    free(xTx);
    UNPROTECT(5);
    return Result;
}

void _Rank_3(int *Sl, int *Sr, double *Y, int n, int *rij, int **R, int *ri)
{
    // input: 
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    // rij in R^{n(n-1)/2} is the upper triangle of Rij without diagonal, by row.
    // Rij is the matrix of partial rank R_{i,j}, 
    // only saves the upper triangle without diagonal, saved by row.
    // ri = sum_{j \ne i} R_{i, j}
    unsigned int i, j, c;
    c = 1;
    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            if ((Sr[i] == 0) || (Sl[j] == 0) || (Y[i] == Y[j])) {
                rij[c] = 0;
            } else {
                rij[c] = -1;
            }
            c++;
        }
    }
    c = 0;
    for (i = 0; i < n-1; ++i) {
        R[i] = rij + c;
        c += n-2-i;
    }
    // remember that R[i][j] + R[j][i] = 0
    for (i = 0; i < n; ++i) {
        ri[i] = 0;
        for (j = 0; j < i; ++j)   ri[i] -= R[j][i];
        for (j = i+1; j < n; ++j) ri[i] += R[i][j];
    }
}

double _spr_test_dc(double **xTx, int *Sl, int *Sr, double *Y, int n, 
    double *sigmaR)
{   
    // input: 
    // xTx in R^{n*n}, is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    int   *ri;
    int   *rank;
    int   **R;
    rank    = (int*)malloc(sizeof(int) * (n*n-n+2)/2);
    ri     = (int*)malloc(sizeof(int) *n);
    R      = (int**)malloc(sizeof(int*)*(n-1));
    _Rank_3(Sl, Sr, Y, n, rank, R, ri);

    // printarray(ri, n);

    unsigned int i, j, k;
    int ell;
    double Test = 0.0;
    unsigned int tmp = (n-2)*(n-3);

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            ell = (ri[i] - R[i][j])*(ri[j] + R[i][j]);
            for (k = 0; k < i; ++k)   ell -= R[k][i] * R[k][j];
            for (k = i+1; k < j; ++k) ell += R[i][k] * R[k][j];
            for (k = j+1; k < n; ++k) ell -= R[i][k] * R[j][k];
            Test += ell * xTx[i][j] / tmp;
        }
    }

    double temp;
    sigmaR[0] = 0.0;
    for (i = 0; i < n; ++i) {
        temp = ri[i] / (n-1.0);
        sigmaR[0] += temp*temp;
    }
    sigmaR[0] /= n*1.0;

    Test /= n*(n-1)/2.0;

    free(ri);
    free(rank);
    free(R);
    return Test;
}

SEXP _SPR_Test_Double_Censor(SEXP _X, SEXP _Y, SEXP _Sl, SEXP _Sr, SEXP _Param)
{
    int n;
    n = INTEGER(_Param)[0];
    double *y;
    y = REAL(_Y);
    int *sl, *sr;
    sl = INTEGER(_Sl);
    sr = INTEGER(_Sr);

    double *xbyx;
    double **xTx;
    xbyx     =  (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    xTx      = (double**)malloc(sizeof(double*)*(n-1));

    SEXP _Test, _Tr_sigma2, _sigmaR;
    PROTECT(_Test = allocVector(REALSXP, 1));
    PROTECT(_Tr_sigma2 = allocVector(REALSXP, 1));
    PROTECT(_sigmaR = allocVector(REALSXP, 1));

    _Gram(REAL(_X), INTEGER(_Param), xbyx, xTx);
    REAL(_Test)[0] = _spr_test_dc(xTx, sl, sr, y, n, REAL(_sigmaR));
    REAL(_Tr_sigma2)[0] = _tr_sigma2_hat(xTx, n);

    SEXP Result, R_names;
    char *names[3] = {"test", "tr_sigma", "sigmaR"};
    PROTECT(Result    = allocVector(VECSXP,  3));
    PROTECT(R_names   = allocVector(STRSXP,  3));
    SET_STRING_ELT(R_names, 0,  mkChar(names[0]));
    SET_STRING_ELT(R_names, 1,  mkChar(names[1]));
    SET_STRING_ELT(R_names, 2,  mkChar(names[2]));
    SET_VECTOR_ELT(Result, 0, _Test);
    SET_VECTOR_ELT(Result, 1, _Tr_sigma2);
    SET_VECTOR_ELT(Result, 2, _sigmaR);
    setAttrib(Result, R_NamesSymbol, R_names); 

    free(xbyx);
    free(xTx);
    UNPROTECT(5);
    return Result;
}







double _spr_test_dependent(double *ri_double, double **xTx, int *S, double *Y, int n, int *ranktype)
{   
    // input: 
    // xTx in R^{n*n}, is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    int   *ri;
    int   *rank;
    int   **R;
    rank    = (int*)malloc(sizeof(int) * (n*n-n+2)/2);
    ri     = (int*)malloc(sizeof(int) *n);
    R      = (int**)malloc(sizeof(int*)*(n-1));
    if (ranktype[0] == 1) {
        _Rank_1(S, Y, n, rank, R, ri);
    } else if (ranktype[0] == 2) {
        _Rank_2(Y, n, rank, R, ri);
    }

    // printarray(ri, n);

    unsigned int i, j, k;
    int ell;
    double Test = 0.0;
    unsigned int tmp = (n-2)*(n-3);

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            ell = (ri[i] - R[i][j])*(ri[j] + R[i][j]);
            for (k = 0; k < i; ++k)   ell -= R[k][i] * R[k][j];
            for (k = i+1; k < j; ++k) ell += R[i][k] * R[k][j];
            for (k = j+1; k < n; ++k) ell -= R[i][k] * R[j][k];
            Test += ell * xTx[i][j] / tmp;
        }
    }

    for (i = 0; i < n; ++i) {
        ri_double[i] = ri[i]/(n-1.0);
    }

    Test /= n*(n-1)/2.0;

    free(ri);
    free(rank);
    free(R);
    return Test;
}


SEXP _SPR_Test_Dependent_Censor(SEXP _X, SEXP _Y, SEXP _S, SEXP _Param, SEXP _Type)
{
    int n;
    n = INTEGER(_Param)[0];
    double *y;
    y = REAL(_Y);
    int *s, *type;
    s = INTEGER(_S);
    type = INTEGER(_Type);

    double *xbyx;
    double **xTx;
    double *ri;
    xbyx     =  (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    xTx      = (double**)malloc(sizeof(double*)*(n-1));
    ri   = (double*)malloc(sizeof(double) *n);

    SEXP _Test, _Tr_sigma2;
    PROTECT(_Test = allocVector(REALSXP, 1));
    PROTECT(_Tr_sigma2 = allocVector(REALSXP, 1));

    _Gram(REAL(_X), INTEGER(_Param), xbyx, xTx);
    REAL(_Test)[0] = _spr_test_dependent(ri, xTx, s, y, n, type);
    int c = 1;
    for (int i = 0; i < n-1; ++i) {
        for (int j = i+1; j < n; ++j) {
            xbyx[c] *= ri[i]*ri[j];
            c++;
        }
    }
    REAL(_Tr_sigma2)[0] = _tr_sigma2_hat(xTx, n);

    SEXP Result, R_names;
    char *names[2] = {"test", "tr_sigma"};
    PROTECT(Result    = allocVector(VECSXP,  2));
    PROTECT(R_names   = allocVector(STRSXP,  2));
    SET_STRING_ELT(R_names, 0,  mkChar(names[0]));
    SET_STRING_ELT(R_names, 1,  mkChar(names[1]));
    SET_VECTOR_ELT(Result, 0, _Test);
    SET_VECTOR_ELT(Result, 1, _Tr_sigma2);
    setAttrib(Result, R_NamesSymbol, R_names); 

    free(ri);
    free(xbyx);
    free(xTx);
    UNPROTECT(4);
    return Result;
}

SEXP _SPR_Test_Part(SEXP _X, SEXP _Y, SEXP _S, SEXP _Z, SEXP _Param, SEXP _Type)
{
    int n;
    n = INTEGER(_Param)[0];
    double *y, *z;
    y = REAL(_Y);
    z = REAL(_Z);
    int *s, *type;
    s = INTEGER(_S);
    type = INTEGER(_Type);

    double *xbyx;
    double **xTx;
    xbyx     =  (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    xTx      = (double**)malloc(sizeof(double*)*(n-1));

    SEXP _Test, _Tr_sigma2, _sigmaR;
    PROTECT(_Test = allocVector(REALSXP, 1));
    PROTECT(_Tr_sigma2 = allocVector(REALSXP, 1));
    PROTECT(_sigmaR = allocVector(REALSXP, 1));

    _Gram(REAL(_X), INTEGER(_Param), xbyx, xTx);
    REAL(_Test)[0] = _spr_test_part(xTx, s, y, n, type, REAL(_sigmaR), z);
    REAL(_Tr_sigma2)[0] = _tr_sigma2_hat(xTx, n);

    SEXP Result, R_names;
    char *names[3] = {"test", "tr_sigma", "sigmaR"};
    PROTECT(Result    = allocVector(VECSXP,  3));
    PROTECT(R_names   = allocVector(STRSXP,  3));
    SET_STRING_ELT(R_names, 0,  mkChar(names[0]));
    SET_STRING_ELT(R_names, 1,  mkChar(names[1]));
    SET_STRING_ELT(R_names, 2,  mkChar(names[2]));
    SET_VECTOR_ELT(Result, 0, _Test);
    SET_VECTOR_ELT(Result, 1, _Tr_sigma2);
    SET_VECTOR_ELT(Result, 2, _sigmaR);
    setAttrib(Result, R_NamesSymbol, R_names); 

    free(xbyx);
    free(xTx);
    UNPROTECT(5);
    return Result;
}

double _spr_test_part_v2(double *ri, double **xTx, int *S, double *Y, int n, int *ranktype, 
    double *z)
{   
    // input: 
    // xTx in R^{n*n}, is the matrix x^T x, 
    // only saves the upper triangle without diagonal, saved by row.
    // S in R^{n}, status.
    // Y has ascending ordered, Y_i <= Y_j
    double   *rank;
    double   **R;
    rank = (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    R    = (double**)malloc(sizeof(double*)*(n-1));
    if (ranktype[0] == 1) {
        _Rank_part_sigmoid(S, Y, z, n, rank, R, ri);
    } else if (ranktype[0] == 2) {
        _Rank_part_norm(S, Y, z, n, rank, R, ri);
    }

    // printarray(ri, n);

    unsigned int i, j, k;
    double ell;
    double Test = 0.0;
    unsigned int tmp = (n-2)*(n-3);

    for (i = 0; i < n-1; ++i) {
        for (j = i+1; j < n; ++j) {
            ell = (ri[i] - R[i][j])*(ri[j] + R[i][j]);
            for (k = 0; k < i; ++k)   ell -= R[k][i] * R[k][j];
            for (k = i+1; k < j; ++k) ell += R[i][k] * R[k][j];
            for (k = j+1; k < n; ++k) ell -= R[i][k] * R[j][k];
            Test += ell * xTx[i][j] / tmp;
        }
    }

    for (i = 0; i < n; ++i) {
        ri[i] /= n-1.0;
    }

    Test /= n*(n-1)/2.0;

    free(rank);
    free(R);
    return Test;
}

SEXP _SPR_Test_Part_v2(SEXP _X, SEXP _Y, SEXP _S, SEXP _Z, SEXP _Param, SEXP _Type)
{
    int n;
    n = INTEGER(_Param)[0];
    double *y, *z;
    y = REAL(_Y);
    z = REAL(_Z);
    int *s, *type;
    s = INTEGER(_S);
    type = INTEGER(_Type);

    double *xbyx;
    double **xTx;
    double *ri;
    xbyx     =  (double*)malloc(sizeof(double) * (n*n-n+2)/2);
    xTx      = (double**)malloc(sizeof(double*)*(n-1));
    ri   = (double*)malloc(sizeof(double) *n);

    SEXP _Test, _Tr_sigma2;
    PROTECT(_Test = allocVector(REALSXP, 1));
    PROTECT(_Tr_sigma2 = allocVector(REALSXP, 1));

    _Gram(REAL(_X), INTEGER(_Param), xbyx, xTx);
    REAL(_Test)[0] = _spr_test_part_v2(ri, xTx, s, y, n, type, z);
    int c = 1;
    for (int i = 0; i < n-1; ++i) {
        for (int j = i+1; j < n; ++j) {
            xbyx[c] *= ri[i]*ri[j];
            c++;
        }
    }
    REAL(_Tr_sigma2)[0] = _tr_sigma2_hat(xTx, n);

    SEXP Result, R_names;
    char *names[3] = {"test", "tr_sigma"};
    PROTECT(Result    = allocVector(VECSXP,  2));
    PROTECT(R_names   = allocVector(STRSXP,  2));
    SET_STRING_ELT(R_names, 0,  mkChar(names[0]));
    SET_STRING_ELT(R_names, 1,  mkChar(names[1]));
    SET_VECTOR_ELT(Result, 0, _Test);
    SET_VECTOR_ELT(Result, 1, _Tr_sigma2);
    setAttrib(Result, R_NamesSymbol, R_names); 

    free(ri);
    free(xbyx);
    free(xTx);
    UNPROTECT(4);
    return Result;
}






