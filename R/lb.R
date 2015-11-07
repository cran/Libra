#' Linearized Bregman solver for linear, binomial, multinomial models
#'  with lasso, group lasso or column lasso penalty.
#' 
#' Solver for the entire solution path of coefficients for Linear Bregman iteration.
#' 
#' The Linearized Bregman solver computes the whole regularization path
#'  for different types of lasso-penalty for gaussian, binomial and 
#'  multinomial models through iterations. It is the Euler forward 
#'  discretized form of the continuous Bregman Inverse Scale Space 
#'  Differential Inclusion. For binomial models, the response variable y
#'  is assumed to be a vector of two classes which is transformed in to \{1,-1\}.
#'  For the multinomial models, the response variable y can be a vector of k classes
#'  or a n-by-k matrix that each entry is in \{0,1\} with $1$ indicates 
#'  the class. Under all circumstances, two parameters, kappa 
#'  and alpha need to be specified beforehand. The definitions of kappa 
#'  and alpha are the same as that defined in the reference paper. 
#'  Parameter alpha is defined as stepsize and kappa is the damping factor
#'  of the Linearized Bregman Algorithm that is defined in the reference paper.
#'
#' @param X An n-by-p matrix of predictors
#' @param y Response Variable
#' @param kappa The damping factor of the Linearized Bregman Algorithm that is
#'  defined in the reference paper. See details. 
#' @param alpha Parameter in Linearized Bregman algorithm which controls the 
#' step-length of the discretized solver for the Bregman Inverse Scale Space. 
#' See details. 
#' @param family Response type
#' @param group.type There are three kinds of group type. "Column" is only 
#' available for multinomial model.
#' @param index For group models, the index is a vector that determines the 
#' group of the parameters. Parameters of the same group should have equal 
#' value in index. Be careful that multinomial group model default assumes 
#' the variables in same column are in the same group.
#' @param intercept if TRUE, an intercept is included in the model (and not 
#' penalized), otherwise no intercept is included. Default is TRUE.
#' @param normalize if TRUE, each variable is scaled to have L2 norm 
#' square-root n. Default is TRUE.
#' @param iter Number of iterations.
#' @return A "lb" class object is returned. The list contains the call, 
#' the type, the path, the intercept term a0 and value for alpha, kappa, 
#' iter, and meanvalue, scale factor of X, meanx and normx. 
#' @references Ohser, Ruan, Xiong, Yao and Yin, Sparse Recovery via Differential
#'  Inclusions, \url{http://arxiv.org/abs/1406.7728}
#' @author Feng Ruan, Jiechao Xiong and Yuan Yao
#' @keywords regression
#' @examples
#' #Examples in the reference paper
#' library(MASS)
#' n = 200;p = 100;k = 30;sigma = 1
#' Sigma = 1/(3*p)*matrix(rep(1,p^2),p,p)
#' diag(Sigma) = 1
#' A = mvrnorm(n, rep(0, p), Sigma)
#' u_ref = rep(0,p)
#' supp_ref = 1:k
#' u_ref[supp_ref] = rnorm(k)
#' u_ref[supp_ref] = u_ref[supp_ref]+sign(u_ref[supp_ref])
#' b = as.vector(A%*%u_ref + sigma*rnorm(n))
#' kappa = 16
#' alpha = 1/32
#' max_step = 160
#' object <- lb(A,b,kappa,alpha,family="gaussian",group="ungrouped",
#'              intercept=FALSE,normalize=FALSE,iter=max_step)
#' attach(object, warn.conflicts= FALSE)
#' plot((0:max_step)*alpha,c(0,path[,1]),type="l",xlim=c(0,3),
#'      ylim=c(min(path),max(path)),xlab="t",ylab=bquote(beta),
#'      main=bquote(paste("LB ",kappa,"=",.(kappa))))
#' for (j in 2:100){
#'   points((0:max_step)*alpha,c(0,path[,j]),type="l")
#' }
#' detach(object)
#' 
#' 
#' #Diabetes, linear case
#' library(Libra)
#' data(diabetes)
#' attach(diabetes)
#' object <- lb(x,y,100,1e-3,family="gaussian",group="ungrouped",iter=1000)
#' plot(object)
#' 
#' #Simulated data, binomial case
#' X <- matrix(rnorm(1000*256), nrow=1000, ncol=256)
#' alpha <- c(rep(1,50), rep(0,206))
#' y <- 2*as.numeric(runif(1000)<1/(1+exp(-X %*% alpha)))-1
#' group = c(rep(1:5,10),rep(6:108,2))
#' result <- lb(X,y,kappa=5,alpha=1,family="binomial",
#'              intercept=FALSE,normalize = FALSE,iter=500)
#' plot(result)
#' 
#' #Simulated data, multinomial case
#' X <- matrix(rnorm(1000*256), nrow=1000, ncol=256)
#' alpha <- matrix(c(rnorm(50*3), rep(0,206*3)),nrow=3)
#' P <- exp(alpha%*%t(X))
#' P <- scale(P,FALSE,apply(P,2,sum))
#' y <- rep(0,1000)
#' rd <- runif(1000)
#' y[rd<P[1,]] <- 1
#' y[rd>1-P[3,]] <- -1
#' result <- lb(X,y,kappa=5,alpha=0.1,family="multinomial",
#'  group.type="columned",intercept=FALSE,normalize = FALSE,iter=500)
#' plot(result)
#' 

lb <- function(X, y, kappa, alpha, family = c("gaussian", "binomial", "multinomial"),
group.type = c("ungrouped", "grouped", "columned"), index = NA, intercept = TRUE, normalize = TRUE, iter = 1000) {
  family <- match.arg(family)
  group.type <- match.arg(group.type)
  if (!is.matrix(X)) stop("X must be a matrix!")
  if (family!="multinomial"){
    if (!is.vector(y)) stop("y must be a vector unless in multinomial model!")
    if (nrow(X) != length(y)) stop("Number of rows of X must equal the length of y!")
    if (family=="binomial" & any(abs(y)!=1)) stop("y must be in {1,-1}")
  }
  if (family=="multinomial"){
    if (is.vector(y)){
      if(nrow(X) != length(y)) stop("Number of rows of X must equal the length of y!")
      y_unique <- unique(y)
      y = sapply(1:length(y_unique),function(x) as.numeric(y==y_unique[x]))
    }
    else if (is.matrix(y)){
      if(nrow(X) != nrow(y)) stop("Number of rows of X and y must equal!")
      if (any((y!=1)&(y!=0)) || any(rowSums(y)!=1)) stop("y should be indicator matrix!")
    }
    else
      stop("y must be a vector or matrix!")
  }
  
  np <- dim(X)
  n <- np[1]
  p <- np[2]
  one <- rep(1, n)
	if(intercept){
    	meanx <- drop(one %*% X)/n
    	X <- scale(X, meanx, FALSE)
  }else meanx <- rep(0,p)
	if(normalize){
	    normx <- sqrt(drop(one %*% (X^2))/n)
	    X <- scale(X, FALSE, normx)
  }else normx <- rep(1,p)
  
	if (group.type == "ungrouped") {
		if (family == "gaussian")
			object <- LB.lasso(X, y, kappa, alpha, intercept, iter)
		else if (family == "binomial")
			object <- LB.logistic(X, y, kappa, alpha, intercept, iter)
		else if (family == "multinomial")
			object <- LB.multilogistic(X, y, kappa, alpha, intercept, iter)
		else stop("No such family type!")
	} else if (group.type == "grouped") {
		if (!is.vector(index)) stop("Index must be a vector!")
	  if ((length(index) + intercept) != ncol(X))
	    stop("Length of index must be the same as the number of columns of X minus the intercept!")
	  if (family == "multinomial")
			object <- LB.multilogistic.group(X, y, kappa, alpha,index, intercept, iter)
		else if (family == "gaussian")
			object <- LB.lasso.group(X, y, kappa, alpha, index, intercept, iter)
		else if (family == "binomial")
			object <- LB.logistic.group(X, y, kappa, alpha, index, intercept, iter)
		else stop("No such family type!")
	} else if (group.type == "columned") {
		if (family == "multinomial") {
		  object <- LB.multilogistic.column(X, y, kappa, alpha, intercept, iter)
		} else stop("columned version only available for multinomial logistic.")
	} else stop("No such group type.")
  # seperate intercept from path
	if (intercept){
	  if (family != "multinomial"){
  		object$a0 <- object$path[,p+1,drop=TRUE]
	  	object$path <- object$path[,-(p+1),drop=FALSE]
	  }else{
	    object$a0 <- t(sapply(1:iter, function(x) 
	      object$path[[x]][,p+1,drop=FALSE]))
	    object$path <- lapply(1:iter, function(x)
	      object$path[[x]][,-(p+1),drop=FALSE] )
	  }
	}else{
	  if (family == "multinomial"){
	    object$a0 <- matrix(rep(0,iter*ncol(y)),nrow = iter)
	  }else{
  	  object$a0 <- rep(0,iter)
	  }
	}
  # re-scale
  if (family == "multinomial"){
    object$path <- lapply(1:iter, function(x)
      scale(object$path[[x]],FALSE,normx) )
  	if (intercept) object$a0 <- object$a0 - 
  	    t(sapply(1:iter, function(x) object$path[[x]]%*%meanx))
  }else{
    object$path <- scale(object$path, FALSE, normx)
    if (intercept) object$a0 <- object$a0 - object$path%*%meanx
  }
	object$meanx <- meanx
	object$normx <- normx
	object$t <- seq(iter)*alpha
#	fit <- predict(object,X)
#	object$fit <- fit
	return(object)
}

LB.lasso <- function(X, y, kappa, alpha, intercept = TRUE, iter = 20) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept))
	solution <- .C("LB_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(intercept)
	)[[8]]
	path <- t(matrix(solution, ncol = iter))
	object <- list(call = call, type = c("Lasso", "ungrouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.lasso.group <- function(X, y, kappa, alpha, index, intercept = TRUE, iter = 100) {
	intercept <- as.integer(intercept != 0)
	call <- match.call()
	ord <- order(index)
	ord_rev <- order(ord)
	group_size <- as.vector(table(index))
	group_split <- c(0, cumsum(group_size))
	group_split_length <- length(group_split)
	row <- nrow(X)
	col <- ncol(X)
	X <- X[,ord]
	result_r <- vector(length = iter * (col + intercept))
	solution <- .C("LB_group_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(group_split),
		as.integer(group_split_length),
		as.integer(intercept)
	)[[8]]
	path <- t(matrix(solution, ncol = iter))
	path[,1:col] <- path[,1:col][,ord_rev]
	object <- list(call = call, type = c("Lasso", "grouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.logistic <- function(X, y, kappa, alpha, intercept = TRUE, iter = 100) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept))
	solution <- .C("LB_logistic_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(intercept))[[8]]
	path <- t(matrix(solution, ncol = iter))
	object <- list(call = call, type = c("logistic", "ungrouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.logistic.group <- function(X, y, kappa, alpha, index, intercept = TRUE, iter = 100) {
	call <- match.call()
	ord <- order(index)
	ord_rev <- order(ord)
	group_size <- as.vector(table(index))
	group_split <- c(0, cumsum(group_size))
	group_split_length <- length(group_split)
	row <- nrow(X)
	col <- ncol(X)
	X <- X[,ord]
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept))
	solution <- .C("LB_logistic_group_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(group_split),
		as.integer(group_split_length),
		as.integer(intercept)
	)[[8]]
	path <- t(matrix(solution, ncol = iter))
	path[,1:col] <- path[,1:col][,ord_rev]
	object <- list(call = call, type = c("logistic", "grouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.multilogistic <- function(X, y, kappa, alpha, intercept = TRUE, iter = 100) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	category <- ncol(y)
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept) * category)
	solution <- .C("LB_multi_logistic_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.integer(category),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(intercept)
	)[[9]]
	path.multi <- lapply(0:(iter - 1), function(x)
		matrix(solution[(1+x*category*(col+intercept)):((x+1)*category*(col+intercept))], category, col+intercept)
	)
	object <- list(call = call, type = c("multilogistic", "ungrouped"), kappa = kappa, alpha = alpha, path = path.multi, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.multilogistic.column <- function(X, y, kappa, alpha, intercept = TRUE, iter = 100) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	category <- ncol(y)
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept) * category)
	solution <- .C("LB_multi_logistic_column_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.integer(category),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(intercept)
	)[[9]]
	path.multi <- lapply(0:(iter - 1), function(x)
		matrix(solution[(1+x*category*(col+intercept)):((x+1)*category*(col+intercept))], category, col+intercept)
	)
	object <- list(call = call, type = c("multilogistic", "columned"), kappa = kappa, alpha = alpha, path = path.multi, iter = iter)
	class(object) <- "lb"
	return(object)
}

LB.multilogistic.group <- function(X, y, kappa, alpha, index, intercept = TRUE, iter = 100) {
	call <- match.call()
	ord <- order(index)
	ord_rev <- order(ord)
	group_size <- as.vector(table(index))
	group_split <- c(0, cumsum(group_size))
	group_split_length <- length(group_split)
	row <- nrow(X)
	col <- ncol(X)
	X <- X[,ord]
	category <- ncol(y)
	intercept <- as.integer(intercept != 0)
	result_r <- vector(length = iter * (col + intercept) * category)
	solution <- .C("LB_multi_logistic_group_lasso",
		as.numeric(X),
		as.integer(row),
		as.integer(col),
		as.numeric(y),
		as.integer(category),
		as.numeric(kappa),
		as.numeric(alpha),
		as.integer(iter),
		as.numeric(result_r),
		as.integer(group_split),
		as.integer(group_split_length),
		as.integer(intercept)
	)[[9]]
	path.multi <- lapply(0:(iter-1), function(x)
		matrix(solution[(1+x*category*(col+intercept)):((x+1)*category*(col+intercept))], category, col+intercept)
	)
	for (i in 1:iter){
	  path.multi[[i]][,1:col] <- path.multi[[i]][,1:col][,ord_rev]
	}
	object <- list(call = call, type = c("multilogistic", "grouped"), kappa = kappa, alpha = alpha, path = path.multi, iter = iter)
	class(object) <- "lb"
	return(object)
}

.onAttach = function(libname, pkgname) {
   packageStartupMessage("Loaded Libra ", as.character(packageDescription("Libra")[["Version"]]), "\n")
}