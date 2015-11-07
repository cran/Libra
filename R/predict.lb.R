#' Predict method for lb objects
#'
#' Predict response variable for new data given a lb object
#' 
#' The default plot uses the fraction of L1 norm as the x. 
#' For multinomial case, the sum of absolute values of different class's 
#' coefficients are caculated to represent each variable.
#' The intercept term is not ploted
#' 
#' @param object lb object
#' @param newx New data matrix that each row is a data or a vector. If missing,
#'    type switched to coefficients
#' @param t The parmeter for object to determin which coeffiecients used for prediction.
#'  Linear interpolation is used if t is not in object\$t. 
#'  If missing, all the coeffiecients along the path is used to predict.
#' @param type To predict response of newx or just fit coeffients on the path.
#' @param \dots Additonal arguments for generic predict.
#' @return A list containing t and other variables. For type="fit", the rediction response
#' "fit" is returned. For "binomial", a vector of the probabilities for newx 
#' falling into class +1 is redurned. For "multinomial", a matrix with each column means
#' the probabilities for newx falling into the corresponding class. If type="coefficients"
#' coefficients "beta" and intercepts "a0" are returned.
#' @author Feng Ruan, Jiechao Xiong and Yuan Yao
#' @keywords methods
#'

predict.lb <-
function(object, newx, t, type = c("fit", "coefficients"),...)
{
  type <- match.arg(type)
  if(missing(newx) & type == "fit") {
    warning("Type=fit with no newx argument; type switched to coefficients")
    type <- "coefficients"
  }
  path <- object$path
  a0 <- object$a0
  if(missing(t)) {
    t <- object$t
    newbetas <- path
    newa0 <- a0
  }else{
    t0 <- object$t
    t[t<min(t0)] <- min(t0)
    t[t>max(t0)] <- max(t0)
    coord <- approx(t0, seq(t0), t)$y
    left <- floor(coord)
    right <- ceiling(coord)
    cright <- (t - t0[left])/(t0[right] - t0[left])
    cleft <- (t0[right] - t)/(t0[right] - t0[left])
    if (object$type[1]!= "multilogistic"){
      if (object$kappa == Inf){
        newbetas <- path[left,]
        newa0 <- a0[left,]
      }else{
        newbetas <- ( cleft*path[left,  , drop = FALSE] +
                       cright*path[right,  , drop = FALSE])
        newbetas[left == right,  ] <- path[left[left == right], ]
        newa0 <- (cleft* a0[left, drop = FALSE] +
                       cright * a0[right, drop = FALSE])
        newa0[left == right] <- a0[left[left == right]]
        newbetas <- drop(newbetas)
        newa0 <- drop(newa0)
      }
    }else{
      newbetas <- lapply(1:length(t), function(x) 
                    cleft[x]* path[[left[x]]] + cright[x]* path[[right[x]]])
      newbetas[left == right] <- path[left[left == right]]
      newa0 <- (cleft * a0[left, , drop = FALSE] +
                  cright * a0[right, , drop = FALSE])
      newa0[left == right,] <- a0[left[left == right],]
    }
  }
  if (type == "fit"){
    n <- dim(newx)[1]
    if (object$type[1]=="Lasso")
      predict <- newx%*%t(newbetas) + rep(1,n)%*%t(newa0)
    else if (object$type[1]=="logistic")
      predict <- 1/(1+exp(-newx%*%t(newbetas) - rep(1,n)%*%t(newa0)))
    else if(object$type[1]=="multilogistic"){
      predict <- lapply(1:length(t),function(x)
        exp(newx%*%t(newbetas[[x]]) + rep(1,n)%*%t(newa0[x,])))
      predict <- lapply(1:length(t),function(x)
       (t(scale(t(predict[[x]]), center=FALSE, scale=rowSums(predict[[x]])))))
    }
  }
  robject <- switch(type,
                    coefficients = list(t=t,betas = newbetas, a0 = newa0),
                    fit = list(t=t,fit=predict))
  robject
}

