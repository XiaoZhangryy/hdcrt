#' High dimensional censored rank partial test
#'
#' @useDynLib hdcrt, .registration = TRUE
#' @export
#' @param x The design matrix.
#' @param y The survival outcome.
#' @param status The right censoring indicator.
#' @param z The control vector, which is the control factors multiplied by the estimated coefficients.
#' @param smooth The smooth function used. "sigmoid" represents the sigmoid prime kernel function and "gaussian" represents the gaussian kernel function.
#' @param h The bandwidth of the control vector. The default value is the standard deviation of z divided by the square root of sample size n.
#' @param covariate_dependence An indicator of whether the censoring mechanism is dependent on covariates.
#'
#' @return A list.
#' \itemize{
#'     \item ts - Test statistic.
#'     \item pval - p value.
#' }
#' @seealso \code{\link{hdcrt}}, \code{\link{hdcrt_dc}}
#' @importFrom stats pnorm var
#' @examples
#' set.seed(0)
#' n <- 150
#' p <- 550
#' x <- matrix(rnorm(n * p), n, p)
#' u <- matrix(rnorm(n * p), n, p)
#' alpha <- c(rep(1, 5), rep(0, p - 5))
#' beta <- c(rep(1, 5), rep(0, p - 5))
#' y <-  u %*% alpha + x %*% beta + rnorm(n)
#' status <- sample(c(0, 1), n, replace = TRUE, prob = c(0.2, 0.8))
#' fit <- sprfabs(y, u, status)
#' alphahat <- fit$opttheta
#' z <- u %*% alphahat
#' test_result_sigmoid <- hdcrpt(x, y, status, z, "sigmoid")
#' print(test_result_sigmoid)
#' test_result_gaussian <- hdcrpt(x, y, status, z, "gaussian")
#' print(test_result_gaussian)
hdcrpt <- function(x, y, status, z, smooth = c("sigmoid", "gaussian"), h = NULL, covariate_dependence = TRUE)
{
    param   <- dim(x)
    y.order <- order(y)
    y       <- y[y.order]
    x       <- x[y.order, ]
    status  <- status[y.order]
    z       <- z[y.order]
    n <- length(y)

    if (is.null(h)) {
        h <- sqrt(var(z) / n)
    }
    z <- z / h

    smooth <- match.arg(smooth)
    if (covariate_dependence) {
        if (smooth == "sigmoid") {
            fit <- .Call("_SPR_Test_Part_v2",
                    as.numeric(t(x)),
                    as.numeric(y),
                    as.integer(status),
                    as.numeric(t(z)),
                    as.integer(param),
                    as.integer(1))
        } else {
            fit <- .Call("_SPR_Test_Part_v2",
                    as.numeric(t(x)),
                    as.numeric(y),
                    as.integer(status),
                    as.numeric(t(z)),
                    as.integer(param),
                    as.integer(2))
        }
        tmp = sqrt(2*fit$tr_sigma)/n
        Tn    = fit$test / tmp
    } else {
        if (smooth == "sigmoid") {
            fit <- .Call("_SPR_Test_Part",
                    as.numeric(t(x)),
                    as.numeric(y),
                    as.integer(status),
                    as.numeric(t(z)),
                    as.integer(param),
                    as.integer(1))
        } else {
            fit <- .Call("_SPR_Test_Part",
                    as.numeric(t(x)),
                    as.numeric(y),
                    as.integer(status),
                    as.numeric(t(z)),
                    as.integer(param),
                    as.integer(2))
        }
        tmp <- sqrt(2 * fit$tr_sigma) / n
        Tn  <- fit$test / (fit$sigmaR * tmp)
    }

    result <- list(
        ts = Tn,
        pvals = 1 - pnorm(Tn)
    )

    return(result)
}


#' High dimensional censored rank test
#'
#' @useDynLib hdcrt, .registration = TRUE
#' @export
#' @param x The design matrix.
#' @param y The survival outcome.
#' @param status The right censoring indicator.
#' @param covariate_dependence An indicator of whether the censoring mechanism is dependent on covariates.
#'
#' @return A list.
#' \itemize{
#'     \item ts - Test statistic.
#'     \item pval - p value.
#' }
#' @seealso \code{\link{hdcrpt}}, \code{\link{hdcrt_dc}}
#' @examples
#' set.seed(0)
#' n <- 150
#' p <- 550
#' x <- matrix(rnorm(n * p), n, p)
#' beta <- c(rep(1, 5), rep(0, p - 5))
#' y <-  x %*% beta + rnorm(n)
#' status <- sample(c(0, 1), n, replace = TRUE, prob = c(0.2, 0.8))
#' test_result <- hdcrt(x, y, status)
#' print(test_result)
hdcrt <- function(x, y, status, covariate_dependence = TRUE)
{
    param   <- dim(x)
    y.order <- order(y)
    n <- length(y)
    y       <- y[y.order]
    x       <- x[y.order, ]
    status  <- status[y.order]

    if (covariate_dependence) {
        fit <- .Call("_SPR_Test_Dependent_Censor",
                as.numeric(t(x)),
                as.numeric(y),
                as.integer(status),
                as.integer(param),
                as.integer(1))
        tmp <- sqrt(2 * fit$tr_sigma) / n
        Tn <- fit$test / tmp
    } else {
        fit <- .Call("_SPR_Test",
                as.numeric(t(x)),
                as.numeric(y),
                as.integer(status),
                as.integer(param),
                as.integer(1))
        if (sum(status) == n) {
            fit$sigmaR <- 1 / 3
        }
        tmp <- sqrt(2 * fit$tr_sigma) / n
        Tn <- fit$test / (fit$sigmaR * tmp)
    }

    result <- list(
        ts = Tn,
        pvals = 1 - pnorm(Tn)
    )

    return(result)
}


#' High dimensional censored rank test with double censored
#'
#' @useDynLib hdcrt, .registration = TRUE
#' @export
#' @param x The design matrix.
#' @param y The survival outcome.
#' @param status_left The left censoring indicator.
#' @param status_right The right censoring indicator.
#'
#' @return A list.
#' \itemize{
#'     \item ts - Test statistic.
#'     \item pval - p value.
#' }
#' @seealso \code{\link{hdcrpt}}, \code{\link{hdcrt}}
#' @examples
#' set.seed(0)
#' n <- 150
#' p <- 550
#' x <- matrix(rnorm(n * p), n, p)
#' beta <- c(rep(1, 5), rep(0, p - 5))
#' y <-  x %*% beta + rnorm(n)
#' status_left <- y >= qnorm(0.1) * sqrt(6)
#' status_right <- y <= qnorm(0.9) * sqrt(6)
#' test_result <- hdcrt_dc(x, y, status_left, status_right)
#' print(test_result)
hdcrt_dc <- function(x, y, status_left, status_right)
{
    message("High dimensional censored rank test with double censored only support independent censoring mechanism for now.")
    param   <- dim(x)
    y.order <- order(y)
    y       <- y[y.order]
    x       <- x[y.order, ]
    status_left  <- status_left[y.order]
    status_right <- status_right[y.order]

    fit <- .Call("_SPR_Test_Double_Censor",
            as.numeric(t(x)),
            as.numeric(y),
            as.integer(status_left),
            as.integer(status_right),
            as.integer(param),
            as.integer(1))

    n <- nrow(x)
    tmp <- sqrt(2 * fit$tr_sigma) / n
    Tn <- fit$test / (fit$sigmaR * tmp)
    
    result <- list(
        ts = Tn,
        pvals = 1 - pnorm(Tn)
    )

    return(result)
}
