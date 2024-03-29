% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ising.R
\name{ising}
\alias{ising}
\title{Linearized Bregman solver for composite conditionally likelihood of Ising model 
 with lasso penalty.}
\usage{
ising(
  X,
  kappa,
  alpha,
  c = 2,
  tlist,
  responses = c(-1, 1),
  nt = 100,
  trate = 100,
  intercept = TRUE,
  print = FALSE
)
}
\arguments{
\item{X}{An n-by-p matrix of variables.}

\item{kappa}{The damping factor of the Linearized Bregman Algorithm that is
defined in the reference paper. See details.}

\item{alpha}{Parameter in Linearized Bregman algorithm which controls the 
step-length of the discretized solver for the Bregman Inverse Scale Space. 
See details.}

\item{c}{Normalized step-length. If alpha is missing, alpha is automatically generated by 
\code{alpha=c*n/(kappa*||X^T*X||_2)}. Default is 2. It should be in (0,4).
If beyond this range the path may be oscillated at large t values.}

\item{tlist}{Parameters t along the path.}

\item{responses}{The type of data. c(0,1) or c(-1,1), Default is c(-1,1).}

\item{nt}{Number of t. Used only if tlist is missing. Default is 100.}

\item{trate}{tmax/tmin. Used only if tlist is missing. Default is 100.}

\item{intercept}{if TRUE, an intercept is included in the model (and not 
penalized), otherwise no intercept is included. Default is TRUE.}

\item{print}{If TRUE, the percentage of finished computation is printed.}
}
\value{
A "ising" class object is returned. The list contains the call, 
 the path, the intercept term a0 and value for alpha, kappa, t.
}
\description{
Solver for the entire solution path of coefficients.
}
\details{
The data matrix X is assumed in \{1,-1\}. The Ising model here used is described as following:\cr
\deqn{P(x) \sim \exp(\sum_i \frac{a_{0i}}{2}x_i + x^T \Theta x/4)}\cr
where \eqn{\Theta} is p-by-p symmetric and 0 on diagnal. Then conditional on \eqn{x_{-j}}\cr
\deqn{\frac{P(x_j=1)}{P(x_j=-1)} = exp(\sum_i a_{0i} + \sum_{i\neq j}\theta_{ji}x_i)}\cr
then the composite conditional likelihood is like this:\cr
\deqn{- \sum_{j} condloglik(X_j | X_{-j})}
}
\examples{

library('Libra')
library('igraph')
data('west10')
X <- as.matrix(2*west10-1);
obj = ising(X,10,0.1,nt=1000,trate=100)
g<-graph.adjacency(obj$path[,,770],mode="undirected",weighted=TRUE)
E(g)[E(g)$weight<0]$color<-"red"
E(g)[E(g)$weight>0]$color<-"green"
V(g)$name<-attributes(west10)$names
plot(g,vertex.shape="rectangle",vertex.size=35,vertex.label=V(g)$name,
edge.width=2*abs(E(g)$weight),main="Ising Model (LB): sparsity=0.51")
}
\author{
Jiechao Xiong
}
\keyword{regression}
