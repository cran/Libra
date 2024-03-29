% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cv.iss.R
\name{cv.iss}
\alias{cv.iss}
\title{CV for ISS}
\usage{
cv.iss(
  X,
  y,
  K = 5,
  t,
  intercept = TRUE,
  normalize = TRUE,
  plot.it = TRUE,
  se = TRUE,
  ...
)
}
\arguments{
\item{X}{An n-by-p matrix of predictors}

\item{y}{Response Variable}

\item{K}{Folds number for CV. Default is 5.}

\item{t}{A vector of predecided tuning parameter.}

\item{intercept}{If TRUE, an intercept is included in the model (and not 
penalized), otherwise no intercept is included. Default is TRUE.}

\item{normalize}{if TRUE, each variable is scaled to have L2 norm 
square-root n. Default is TRUE.}

\item{plot.it}{Plot it? Default is TRUE}

\item{se}{Include standard error bands? Default is TRUE}

\item{\dots}{Additonal arguments passing to lb}
}
\value{
A list is returned. The list contains a vector of parameter t, 
crossvalidation error cv.error, and the estimated standard deviation for it cv.sd
}
\description{
Cross-validation method to tuning the parameter t for ISS.
}
\details{
K-fold cross-validation method is used to tuning the parameter $t$ for ISS.
Mean square error is used as prediction error.
}
\examples{
#Examples in the reference paper
library(MASS)
n = 200;p = 100;k = 30;sigma = 1
Sigma = 1/(3*p)*matrix(rep(1,p^2),p,p)
diag(Sigma) = 1
A = mvrnorm(n, rep(0, p), Sigma)
u_ref = rep(0,p)
supp_ref = 1:k
u_ref[supp_ref] = rnorm(k)
u_ref[supp_ref] = u_ref[supp_ref]+sign(u_ref[supp_ref])
b = as.vector(A\%*\%u_ref + sigma*rnorm(n))
cv.iss(A,b,intercept = FALSE,normalize = FALSE)

}
\references{
Ohser, Ruan, Xiong, Yao and Yin, Sparse Recovery via Differential
 Inclusions, \url{https://arxiv.org/abs/1406.7728}
}
\author{
Feng Ruan, Jiechao Xiong and Yuan Yao
}
\keyword{Cross-validation}
