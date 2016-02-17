\name{isingdata}
\alias{isingdata}

\title{
Simulation data for Ising model
}
\description{
The \code{isingdata} data list contains 2 variables. One is 5000 samples from the ising model on the 10-by-10 grid using Gibbs sampling. The other is the  groupdtruth parameter.
}
\format{
  This data list contains the following two variables:
  \describe{
    \item{J}{a 100-by-100 matrix, the groudtruth of ising model.}
    \item{X}{a 5000-by-100 matrix, each entry is in {-1,1}.}
  }
}
\keyword{datasets}