#include <R.h>
#include <Rinternals.h>
#include <math.h>
#include <Rmath.h>
#include <stdio.h>
#include <stdlib.h>

// loss function for spr model
// Assuming y is already sorted in ascending order
// do not consider tide, update later when necessary
double spr_loss(double *y, int *status, double *xb, double sigma, int n)
{
    int i, j; 
    double s = -1.0 / sigma;

    double corr = 0.0;
    for (i=0; i<n; i++){
         for (j=0; j<i; j++){
            if (status[j] == 0) continue;
         	corr += 1.0 / (1 + exp(s * (xb[i] - xb[j])));	 
		}
    }

    return corr;
}

// update sparse index, save beta into a sparse matrix formula
// rows are covariances, columns are iteration counts
void usi(int *s_i, int *s_j, int *tt_b, int *tt_a, int *act, int ns, int iter)
{
    /*
    s_i - pointer to rows of sparse matrix
    s_j - pointer to columns of sparse matrix
    tt_b - update to end point of last record
    tt_a - end point of last record, update to start point of next record
    act - activate set
    ns - cardinality of activate set
    iter - current iteration count
    */
    int i;
    s_i += *tt_a;
    s_j += *tt_a;
    for (i = 0; i < ns; ++i)
    {
        s_i[i] = act[i];
        s_j[i] = iter;
    }
    *tt_b = *tt_a;
    *tt_a += ns;
}

// derivative function for spr model
// Assuming y is already sorted in ascending order
// do not consider tide, update later when necessary
void spr_der(double *y, double *x, double *xb, int *status, double sigma, int n, int p, double *derivative)
{
    /*
    y - n-vector, pointer of response
    x - n*p-vector, pointer of predictors
    xb - n-vector, pointer of linear operator, xb = X^T beta
    status - n-vector, pointer of status
    n - length of y and xb
    p - number of predictors
    derivative - p-vector, pointer of derivative
    */ 

	int i, j, k;
    double u;
    double s = -1.0 / sigma;
    
    // derivative doesn't need initial value
    for (i = 0; i < p; ++i) derivative[i] = 0.0;

    for (i=0; i<n; i++){
        for (j=0; j<i; j++){ 
            if (status[j] == 0) continue;
            u =  (xb[i] - xb[j])*s;

            if (u < -500) {
                continue; // u = 0;
            } else {
                u = exp(u);
                u = u/pow(1.0 + u, 2.0)/sigma;
            }

            for (k=0; k<p; k++) {
                derivative[k] += u * (x[i+k*n] - x[j+k*n]);
            }
        }
    }  

    u = -1.0 / (n * (n - 1.0));
    for (i = 0; i < p; ++i) derivative[i] *= u;
}

// bic compution
double spr_bic(double loss, int *param)
{
    int n = param[0];
    int number = param[2];
    int sum_status = param[3];
    return log(-loss) - number * log(sum_status)/2.0/n;
}

// take an initial step
void spr_Initial(double *y, double *x, double *xb, double sigma, int *param, double *weight, double eps, double *loss, double *lambda, 
            int *direction, int *active, double *bic, int *df, double *derivative, double *beta, double *residual, int *status)
{
    int i, j, n = *param, p = param[1], count = 0;
    double temp, value, *x_j;

    // calculate the derivative
    spr_der(y, x, xb, status, sigma, n, p, derivative);

    // find a forward direction
    temp = fabs(*derivative)/weight[0];
    for(j=1; j<p; j++){
		value = fabs(derivative[j])/weight[j];
		if(value > temp){
			temp    = value;
			count   = j;
		}
	}
    // calculate increment, update xb, lambda, beta, loss, direction, bic, df, param[2] (number of nonzero elements)
    value = eps/weight[count];
    if(derivative[count] > 0.0){value *= -1.0;}
    *beta = value;

    temp    = -1.0*spr_loss(y, status, xb, sigma, n);
    x_j     = x + count*n;
    for(i=0; i<n; i++){
        xb[i] += x_j[i]*value;
	}
    value       = -1.0*spr_loss(y, status, xb, sigma, n);
    *loss       = value;
    *lambda     = (temp - value)/eps;
    *direction  = 1;
    param[2]    = 1; // param[0] = n; param[1] = p; param[2] = active
    *active     = count;
    *bic        = spr_bic(value, param);
    *df         = 1;
}

// try a backward step
// penalized on the intercept term
int spr_Backward(double *y, double *x, double *xb, double sigma, int *param, double *weight, double epsilon, double xi, double *loss, double *lambda, 
            int *direction, int *active, double *bic, int *df, double *derivative, double *beta, double *betanew, double *residual, int *status)
{
    int index, l, count = 0, n = *param, p = param[1], number = param[2], c = -1;
    double dactive, value, *x_j, temp;

    // calculate the derivative
    spr_der(y, x, xb, status, sigma, n, p, derivative);
    // try a backward step by sort
    value = 1e14;
    for(l=0; l<number; l++){
        betanew[l]  = beta[l];
        index       = active[l];
        dactive     = derivative[index]/weight[index];
        if(beta[l] > 0.0){dactive *= -1.0;}
        if(dactive < value){value = dactive; count = index; c = l;}
    }
    value   = epsilon/weight[count];
    temp    = beta[c];
    if (temp > 0) value *= -1.0;
    betanew[c] += value;
    x_j         = x + count*n;
    for(index=0; index<n; index++){
        residual[index] = xb[index] + x_j[index]*value;
    }
    // calculate the object function and determine if we take a backward step
    value = -1.0*spr_loss(y, status, residual, sigma, n);
    if (value - *loss - (*lambda)*epsilon/weight[count] < -1.0*xi){
        // adopt a backward step;
        // update xb, lambda, beta, loss, direction, bic, df, param[2] (number of nonzero elements)
        loss[1]         = value;
        direction[1]    = -1;
        df[1]           = *df;
        lambda[1]       = *lambda;
        for (index = 0; index < n; ++index)
            xb[index] = residual[index];
        // test if vanish
        if(fabs(betanew[c]) < xi){
            for(l=c; l<number-1; l++){ 
                active[l]   = active[l+1];
                betanew[l]  = betanew[l+1];
            }
            param[2]--;
            df[1]--;
        }
        bic[1] = spr_bic(value, param);
        return 0;
    }

    betanew[c] = temp;
    return 1;
}

// take a forward step
void spr_Forward(double *y, double *x, double *xb, double sigma, int *param, double *weight, double epsilon, double *loss, double *lambda, 
            int *direction, int *active, double *bic, int *df, double *derivative, double *beta, int *status)
{
    int i, j, l, n = *param, p = param[1], number = param[2], count = 0, c;
    double temp, value, *x_j;

    // d has calculated in Backward
    // beta has been assigned in backward
    // find a forward direction
    temp = fabs(*derivative)/weight[0];
    for(j=1; j<p; j++){
        value = fabs(derivative[j])/weight[j];
        if(value > temp){
            temp    = value;
            count   = j;
        }
    }
    // calculate increment, update beta, active, param[2], xb, lambda, loss, direction, bic, df
    value = epsilon/weight[count];
    if(derivative[count] > 0.0){value *= -1.0;}
    c = 0;
    for(l=0; l<number; l++){
        if(count == active[l]){
            c++;
            beta[l] += value;
            df[1]   = *df;
            break;}
    }
    if(c==0){
        active[number]  = count;
        beta[number]    = value;
        param[2]        += 1;
        df[1]           = *df + 1;
    }
    x_j = x + count*n;
    for(i=0; i<n; i++){
        xb[i] += x_j[i]*value;
    }
    value           = -1.0*spr_loss(y, status, xb, sigma, n);
    loss[1]         = value;
    bic[1]          = spr_bic(value, param);
    temp            = (*loss - value)/epsilon;
    lambda[1]       = temp < *lambda ? temp : *lambda;
    direction[1]    = 1;
}

// estimate function
int spr_EST(double *y, double *x, double *xb, double sigma, int *param, double *weight, double epsilon, double xi, int *max_iter, double *loss, double *lambda, int *direction, 
                int *active, double *bic, int *df, double *derivative, int *sparse_i, int *sparse_j, double lam_m, int max_s, double *beta, double *residual, int *status, int message)
{
    int k, l;
    double temp;

    // step 1: initial step (forward)
    spr_Initial(y, x, xb, sigma, param, weight, epsilon, loss, lambda, direction, active, bic, df, derivative, beta, residual, status);
    int tt_act_b = 0;
    int tt_act_a = 0;
    usi(sparse_i, sparse_j, &tt_act_b, &tt_act_a, active, param[2], 0);

    // step 2: forward and backward
    for(k=0; k<*max_iter-1; k++){
        l = spr_Backward(y, x, xb, sigma, param, weight, epsilon, xi, loss+k, lambda+k, 
                    direction+k, active, bic+k, df+k, derivative, beta+tt_act_b, beta+tt_act_a, residual, status);
        if(l){
            spr_Forward(y, x, xb, sigma, param, weight, epsilon, loss+k, lambda+k, direction+k, active, bic+k, df+k, derivative, beta+tt_act_a, status);
        }
        usi(sparse_i, sparse_j, &tt_act_b, &tt_act_a, active, param[2], k+1);
        temp = lambda[k+1];
        if (temp <= lambda[0] * lam_m ) {
            *max_iter = k+2;
            if (temp < 0) 
            {
                (*max_iter)--;
                tt_act_a -= param[2];
            }
            break;
        }
        if (param[2] > max_s) {
            if (message == 1) {
                Rprintf("Warning! Max nonzero number is larger than predetermined threshold. Program ended early.\n");
            }
            *max_iter = k+2;
            break;
        }
        if (k == *max_iter-2) {
            if (message == 1) {
                Rprintf("Solution path unfinished, more iterations are needed.\n");
            }
            break;
        }
    }

	return tt_act_a;
}

SEXP _SPRFABS(SEXP Y, SEXP X, SEXP Status, SEXP WEIGHT, SEXP PARAM, SEXP ITER, SEXP Max_S, SEXP PARA_DOUBLE_, SEXP MESSAGE_)
{
    // X has been scaled in R
    double *xb, *loss, *lambda, *bic, *derivative, *beta, *residual;
    int *direction, *active, *df, *sparse_i, *sparse_j;
	int i, tt_act_a;
    int message = INTEGER(MESSAGE_)[0];

	int *para 	= INTEGER(PARAM);
	int n     	= para[0];
	int p     	= para[1];
	int iter	= INTEGER(ITER)[0];
    int max_s   = INTEGER(Max_S)[0];

	double *para1 	= REAL(PARA_DOUBLE_);
	double eps    	= para1[0];
	double xi		= para1[1];
    double lam_m    = para1[2];
    double sigma    = para1[3];

    xb          = (double*)calloc(n, sizeof(double)); // x^T beta, needs initialized
    loss        = (double*)malloc(sizeof(double)*iter);
    lambda      = (double*)malloc(sizeof(double)*iter);
    direction   =    (int*)malloc(sizeof(int)   *iter);
    active      = (int*)malloc(sizeof(int) *(max_s+2));
    bic         = (double*)malloc(sizeof(double)*iter);
    df          =    (int*)malloc(sizeof(int)   *iter);
    derivative  = (double*)malloc(sizeof(double)  *p);
    sparse_i    =    (int*)calloc(iter*max_s, sizeof(int));
    sparse_j    =    (int*)calloc(iter*max_s, sizeof(int));
    beta        = (double*)malloc(sizeof(double)*iter*max_s);
    residual    = (double*)malloc(sizeof(double)  *n);

    tt_act_a = spr_EST(REAL(Y), REAL(X), xb, sigma, para, REAL(WEIGHT), eps, xi, &iter, loss, lambda, direction, active, bic, df, 
                            derivative, sparse_i, sparse_j, lam_m, max_s, beta, residual, INTEGER(Status), message);

    SEXP Beta, Lambda, Direction, Loops, BIC, Loss, Df, Indexi, Indexj;
    SEXP Result, R_names;
    char *names[9] = {"beta", "lambda", "direction", "iter", "bic", "loss", 
    "df", "index_i", "index_j"};
    PROTECT(Beta      = allocVector(REALSXP, tt_act_a));
    PROTECT(Indexi    = allocVector(INTSXP,  tt_act_a));
    PROTECT(Indexj    = allocVector(INTSXP,  tt_act_a));
    PROTECT(Lambda    = allocVector(REALSXP, iter));
    PROTECT(BIC       = allocVector(REALSXP, iter));
    PROTECT(Loss      = allocVector(REALSXP, iter));
    PROTECT(Direction = allocVector(INTSXP,  iter));
    PROTECT(Df        = allocVector(INTSXP,  iter));
    PROTECT(Loops     = allocVector(INTSXP,  1));
    PROTECT(Result    = allocVector(VECSXP,  9));
    PROTECT(R_names   = allocVector(STRSXP,  9));

    for(i = 0; i < 9; ++i) SET_STRING_ELT(R_names, i,  mkChar(names[i]));
    INTEGER(Loops)[0] = iter;
    for (i = 0; i < tt_act_a; ++i) 
    {
        REAL(Beta)[i]       = beta[i];
        INTEGER(Indexi)[i]  = sparse_i[i];
        INTEGER(Indexj)[i]  = sparse_j[i];
    }
    for (i = 0; i < iter; ++i) 
    {
        REAL(BIC)[i]          = bic[i];
        REAL(Loss)[i]         = loss[i];
        REAL(Lambda)[i]       = lambda[i];
        INTEGER(Direction)[i] = direction[i];
        INTEGER(Df)[i]        = df[i];
    }
    
    SET_VECTOR_ELT(Result, 0, Beta);
    SET_VECTOR_ELT(Result, 1, Lambda);
    SET_VECTOR_ELT(Result, 2, Direction);
    SET_VECTOR_ELT(Result, 3, Loops); 
    SET_VECTOR_ELT(Result, 4, BIC);
    SET_VECTOR_ELT(Result, 5, Loss);  
    SET_VECTOR_ELT(Result, 6, Df);  
    SET_VECTOR_ELT(Result, 7, Indexi);  
    SET_VECTOR_ELT(Result, 8, Indexj);    
    setAttrib(Result, R_NamesSymbol, R_names); 


    free(xb);
    free(loss);
    free(lambda);
    free(direction);
    free(active);
    free(bic);
    free(df);
    free(derivative);
    free(sparse_i);
    free(sparse_j);
    free(beta);
    free(residual);

	UNPROTECT(11);
	return Result;
}