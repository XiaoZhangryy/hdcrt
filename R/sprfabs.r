#' A forward and backward stagewise algorithm for high-dimensional spr problem.
#'
#' @useDynLib hdcrt, .registration = TRUE
#' @export
#' @param y The response.
#' @param x The design matrix.
#' @param status The right censoring indicator.
#' @param eps The step size for updating coefficients. Default is \code{eps = 0.01}.
#' @param xi The threshold for qfabs. Default is \code{xi = 1e-10}.
#' @param maxIter The maximum number of outer-loop iterations allowed. Default is \code{maxIter = 3000}.
#' @param sigma The tuning parameter in the Sigmoid function. Default is \code{NULL}.
#' @param weight An optional weights. Default is 1 for each observation.
#' @param nmax Limit the maximum number of variables in the model. When exceed this limit, program will early stopped. Default is \code{NULL}.
#' @param lam_m The ratio of the minumum lambda and the maximum lambda. Default is \code{NULL}.
#' @param message An indicator of whether print warning messages.
#'
#' @return A list.
#' \itemize{
#'   \item theta - The estimation of covariates.
#'   \item opttheta - The optimal estimation of covariates.
#'   \item lambda - lambda sequence.
#'   \item direction - direction sequence.
#'   \item iter - Iterations.
#'   \item bic - The EBIC for each solution.
#'   \item loss - loss for each solution.
#'   \item df - Number of nonzero coefficients.
#'   \item opt - Position of the optimal lambda based on EBIC.
#' }
#'
#' @importFrom Matrix sparseMatrix t
#' @examples
#' set.seed(0)
#' n <- 150
#' p <- 550
#' x <- matrix(rnorm(n * p), n, p)
#' alpha <- c(rep(1, 5), rep(0, p - 5))
#' y <-  x %*% alpha + rnorm(n)
#' status <- sample(c(0, 1), n, replace = TRUE, prob = c(0.2, 0.8))
#' fit <- sprfabs(y, x, status)
#' alphahat <- as.vector(fit$opttheta)
#' print(which(alphahat != 0))
#' print(alphahat[alphahat != 0])
sprfabs <- function(y, x, status, eps = 0.01, xi = 1e-10, maxIter = 3000, sigma = NULL, weight = NULL, nmax = NULL, lam_m = NULL, message = TRUE){

	n 	= length(y)
	p 	= ifelse(is.null(ncol(x)) , 1, ncol(x))

    y.order = order(y)
    x   = x[y.order, ]
    y   = y[y.order]
    status  = status[y.order]

    if(is.null(sigma)) sigma = 1/sqrt(n)
	if(is.null(weight)) weight = rep(1, p)
	if(is.null(nmax)){
		nmax = min(p, n/log(n))
	}
	else if(! is.numeric(nmax)){
		nmax = min(p, n/log(n))
	}
	else if(nmax == 0){
		nmax = min(p, n/log(n))
	}
	else{
		if( nmax > min(n/2, p)){
			warning("The specified number of nonzero coefficients is too large. It is segguested that nmax = min(p, n/log(n)).")
		}
	}
    if (is.null(lam_m)) lam_m = {if (n > p) 1e-4 else .02}

	result 	= standardize(x)
	xx 		= result$xx
    center  = result$center
    scale   = result$scale

    param_int 		= c(n, p, 0, sum(status))
    param_double 	= c(eps, xi, lam_m, sigma)
    fit <- .Call("_SPRFABS",
                as.numeric(y),
                as.numeric(xx),
                as.integer(status),
				as.numeric(weight),
                as.integer(param_int),
                as.integer(maxIter),
                as.integer(nmax),
                as.numeric(param_double),
                as.integer(message)
                )
    opt      	= which.max(fit$bic)
	theta 		= sparseMatrix(fit$index_i, fit$index_j, x = fit$beta, dims = c(p, fit$iter), index1 = FALSE)
	theta 		= theta/matrix(rep(scale,fit$iter),p, fit$iter)
	# theta[1,]    = theta[1,] - center[-1] %*% theta[-1,]
	opttheta 	= theta[,opt,drop=FALSE]

	val = list(	theta     = theta,
                opttheta  = opttheta,
                lambda    = fit$lambda,
                direction = fit$direction,
                iter      = fit$iter,
                bic       = fit$bic,
                loss      = fit$loss,
                df        = fit$df,
                opt       = opt
                )

	return(val)
}

standardize = function(x)
{
   p = ncol(x)
   n = nrow(x)
   center = colMeans(x)
   x.mean = x - matrix(rep(center, n), n, p, byrow=T)
   scale = sqrt(colSums(x.mean^2)/n)
   scale[which(scale==0)] = 1
   center[which(scale==0)] = 0
   xx = t(t(x.mean)/scale)
   list(xx = xx, center = center, scale = scale)
}

