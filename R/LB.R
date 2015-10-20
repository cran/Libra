LB <- function(X, y, kappa, alpha, family = c("gaussian", "binomial", "multinomial"),
group.type = c("ungrouped", "grouped", "blocked"), index = NA, intercept = TRUE, normalize = TRUE, iter = 100) {
	if (!is.matrix(X) || !is.vector(y)) stop("X must be a matrix and y must be a vector!")
	if (nrow(X) != length(y)) stop("Number of rows of X must equal the length of y!")
	family <- match.arg(family)
	group.type <- match.arg(group.type)
	if(normalize){
		n <- nrow(X)
	    normx <- sqrt(drop(rep(1,n) %*% (X^2))/n)
    	X <- scale(X, FALSE, normx)	# scales X
  	}else normx = rep(1,ncol(X))
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
		if (family == "multinomial")
			object <- LB.multilogistic.group(X, y, kappa, alpha, intercept, iter)
		else {
			if ((length(index) + intercept) != ncol(X))
				stop("Length of index must be the same as the number of columns of X minus the intercept!")
			if (family == "gaussian")
				object <- LB.lasso.group(X, y, kappa, alpha, index, intercept, iter)
			else if (family == "binomial")
				object <- LB.logistic.group(X, y, kappa, alpha, index, intercept, iter)
			else stop("No such family type!")
		}
	} else if (group.type == "blocked") {
		if (family == "multinomial") {
			if ((length(index) + intercept) != ncol(X))
				stop("Length of index must be the same as the number of columns of X minus the intercept!")
			object <- LB.multilogistic.block(X, y, kappa, alpha, index, intercept, iter)
		} else stop("Blocked version only available for multinomial logistic.")
	} else stop("No such group type.")
	object$intercept = intercept
	object$normalize = normalize
	object$normx = normx
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
	class(object) <- "LB"
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
	class(object) <- "LB"
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
	class(object) <- "LB"
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
	class(object) <- "LB"
	return(object)
}

LB.multilogistic <- function(X, y, kappa, alpha, intercept = TRUE, iter = 100) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	y_unique <- unique(y)
	category <- length(y_unique)
	for (i in 1:length(y)) y[i] <- which(y_unique == y[i])
	y <- y - 1
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
	path <- t(sapply(1:iter, function(x) colSums(abs(path.multi[[x]]))))
	object <- list(call = call, type = c("multilogistic", "ungrouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "LB"
	return(object)
}

LB.multilogistic.group <- function(X, y, kappa, alpha, intercept = TRUE, iter = 100) {
	call <- match.call()
	row <- nrow(X)
	col <- ncol(X)
	y_unique <- unique(y)
	category <- length(y_unique)
	for (i in 1:length(y)) y[i] <- which(y_unique == y[i])
	y <- y - 1
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
		as.integer(intercept)
	)[[9]]
	path.multi <- lapply(0:(iter - 1), function(x)
		matrix(solution[(1+x*category*(col+intercept)):((x+1)*category*(col+intercept))], category, col+intercept)
	)
	path <- t(sapply(1:iter, function(x) colSums(abs(path.multi[[x]]))))
	object <- list(call = call, type = c("multilogistic", "grouped"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "LB"
	return(object)
}

LB.multilogistic.block <- function(X, y, kappa, alpha, index, intercept = TRUE, iter = 100) {
	call <- match.call()
	ord <- order(index)
	ord_rev <- order(ord)
	group_size <- as.vector(table(index))
	group_split <- c(0, cumsum(group_size))
	group_split_length <- length(group_split)
	row <- nrow(X)
	col <- ncol(X)
	X <- X[,ord]
	y_unique <- unique(y)
	category <- length(y_unique)
	for(i in 1:length(y)) y[i] <- which(y_unique == y[i])
	y <- y - 1
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
	path <- t(sapply(1:iter,function(x) colSums(abs(path.multi[[x]]))))
	path[,1:col] <- path[,1:col][,ord_rev]
	object <- list(call = call, type = c("multilogistic", "blocked"), kappa = kappa, alpha = alpha, path = path, iter = iter)
	class(object) <- "LB"
	return(object)
}

.onAttach = function(libname, pkgname) {
   packageStartupMessage("Loaded Libra ", as.character(packageDescription("Libra")[["Version"]]), "\n")
}