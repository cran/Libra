#LB
#plot 
plot.LB <- function(x,omit.zeros=TRUE,eps= 1e-10, ...){
	coef <- x$path
	if (x$intercept){
		coef = coef[,-ncol(coef)]
	}
	if(omit.zeros){
		c1 <- drop(rep(1, nrow(coef)) %*% abs(coef))
		nonzeros <- c1 > eps
		cnums <- seq(nonzeros)[nonzeros]
		coef <- coef[,nonzeros,drop=FALSE]
	}else {cnums <- seq(ncol(coef))}
	stepid <- seq(nrow(coef))
	s <- apply(abs(coef),1,sum)
	s <- s/max(s)
	matplot(s,coef,xlab="Solution-Path",ylab="Standard Coefficients",xlim = c(0,1),ylim = c(min(coef),max(coef)),pch="*",type="b",...)
	title(paste(x$type[1],x$type[2],sep="-"), line=3)
	abline(h=0, lty=2.5)
	axis(3, at=s, labels=paste(stepid), cex=.5)
	axis(4,at=coef[nrow(coef),],labels=paste(cnums),cex=0.8)
}