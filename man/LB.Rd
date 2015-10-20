\name{LB}
\alias{LB}
\title{
	Linearized Bregman solver for linear, binomial, multinomial models with lasso, group lasso or block lasso penalty
}
\description{
	Solver for the entire solution path of coefficients for Linear Bregman iteration.  
}
\usage{
LB(X,y,kappa,alpha,family=c("gaussian","binomial","multinomial"),
	group.type=c("ungrouped", "grouped", "blocked"),index=NA,intercept=TRUE,
	normalize=TRUE,iter=100)
}
\arguments{
\item{X}{
 	An n-by-p matrix of predictors
}
\item{y}{ 
  	Response Variable
}
\item{kappa}{
  	The damping factor of the Linearized Bregman Algorithm that is defined in the reference paper. See details. 
}
\item{alpha}{
	Parameter in Linearized Bregman algorithm which controls the step-length of the discretized solver for the Bregman Inverse Scale Space. See details. 
 }
\item{family}{
	Response type
}
\item{group.type}{
	There are three kinds of group type. "Block" is only available for multinomial model.
}
\item{index}{
	For group models, the index is a vector that determines the group of the parameters. Parameters of the same group should have equal value in index. Be careful that multinomial block model needs the index to determine the block, while the group model doesn't.
}
\item{intercept}{
	if TRUE, an intercept is included in the model (and not penalized), otherwise no intercept is included. Default is TRUE.
}
\item{normalize}{
	if normalize, each variable is scaled to have L2 norm square-root n. Default is TRUE.
}
\item{iter}{
   Number of iterations.
}
}
\details{
	The Linearized Bregman solver computes the whole regularization path for different types of lasso-penalty for gaussian, binomial and multinomial models through iterations. It is the Euler forward discretized form of the continuous Bregman Inverse Scale Space Differential Inclusion. For binomial models, the classes which the response variable y falls into are assumed to be +1 and -1. For the multinomial models, the classes of the response variable y can be any set of real numbers. Under all circumstances, two parameters, kappa and alpha need to be specified beforehand. The definitions of kappa and alpha are the same as that defined in the reference paper. Parameter alpha is defined as stepsize and kappa is the damping factor of the Linearized Bregman Algorithm that is defined in the reference paper.
}
\value{
	An "LB" class object is returned. The list contains the call, the type, the path and value for alpha, kappa, intercept, normalize and scale factor of X, normx. 
}
\references{
	Ohser, Ruan, Xiong, Yao and Yin, Sparse Recovery via Differential Inclusions, \url{http://arxiv.org/abs/1406.7728}
}
\author{
	Feng Ruan, Jiechao Xiong and Yuan Yao
}
\examples{
#Examples in the reference paper
library(MASS)
library(lars)
n = 200;p = 100;k = 30;sigma = 1
Sigma = 1/(3*p)*matrix(rep(1,p^2),p,p)
diag(Sigma) = 1
A = mvrnorm(n, rep(0, p), Sigma)
u_ref = rep(0,p)
supp_ref = 1:k
u_ref[supp_ref] = rnorm(k)
u_ref[supp_ref] = u_ref[supp_ref]+sign(u_ref[supp_ref])
b = as.vector(A\%*\%u_ref + sigma*rnorm(n))
lasso = lars(A,b,normalize=FALSE,intercept=FALSE,max.steps=100)
attach(lasso)
pdf(file="myplot.pdf")
par(mfrow=c(2,2)) 
plot(n/lambda,beta[1:100,1],type='l',xlim=c(0,3),ylim=c(min(beta),max(beta)),
   xlab='t',ylab=bquote(beta),main='Lasso')
for (i in 2:100){
   points(n/lambda,beta[1:100,i],type='l')
}
kappa_list = c(4,16,64)
alpha_list = 1/2/kappa_list
max_step = floor(10*kappa_list)
for (i in 1:3){
   object <- LB(A,b,kappa_list[i],alpha_list[i],family="gaussian",group="ungrouped",
      intercept=FALSE,normalize=FALSE,iter=max_step[i])
   attach(object, warn.conflicts= FALSE)
   plot((0:max_step[i])*alpha,c(0,path[,1]),type='l',xlim=c(0,3),
   ylim=c(min(path),max(path)),xlab='t',ylab=bquote(beta),
   main=bquote(paste('LB ',kappa,'=',.(kappa_list[i]))))
   for (j in 2:100){
      points((0:max_step[i])*alpha,c(0,path[,j]),type='l')
   }
   detach(object)
}
dev.off()

#Diabetes, linear case
library(Libra)
data(diabetes)
attach(diabetes)
object <- LB(x,y,100,1e-3,family="gaussian",group="ungrouped",iter=1000)
plot(object)

#Simulated data, nonlinear case
X <- matrix(rnorm(1000*256), nrow=1000, ncol=256)
alpha <- c(rep(1,50), rep(0,206))
y <- 2*as.numeric(runif(1000)<1/(1+exp(-X \%*\% alpha)))-1
result <- LB(X,y,kappa=5,alpha=0.1,family="binomial",
	intercept=FALSE,normalize = FALSE,iter=500)
plot(result)
}

\keyword{regression}
\seealso{plot method for LB}
