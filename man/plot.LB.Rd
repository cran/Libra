\name{plot.LB}
\alias{plot.LB}
\title{
	Plot method for LB objects
}
\description{
	Produce a plot of an LB fit. The default is a complete coefficient path.
}
\usage{
\method{plot}{LB}(x,omit.zeros=TRUE,eps= 1e-10, ...)
}
\arguments{
\item{x}{LB object}
\item{omit.zeros}{When the number of variables  is much greater than
    the number of observations, many coefficients will never be nonzero;
      this logical (default \code{TRUE}) avoids plotting these zero coefficents
}
\item{eps}{Definition of zero above, default is \code{1e-10}}
\item{\dots}{Additonal arguments for generic plot. Can be used to set xlims, change colors, line widths, etc
}
}
\author{
	Feng Ruan, Jiechao Xiong and Yuan Yao
}
\details{
The default plot uses the fraction of L1 norm as the x.}
\value{NULL}
\keyword{methods}
