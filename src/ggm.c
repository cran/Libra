//  Created by jiechaoxiong on 16-2-13.
//  Copyright (c) 2016年 jcxiong. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <time.h>
#include "LBheader.h"

void ggm(double* S_r, int* p_r, double* kappa_r, double*alpha_r,  double*result_r,double* t_r,int* nt_r,double* trate_r,int*print)
{
    int iter=0,p=*p_r, nt=*nt_r,k=0;
    double kappa = *kappa_r, alpha=*alpha_r, trate=*trate_r;
    time_t start = clock(),end;
    gsl_matrix *S = gsl_matrix_alloc(p, p);
    read_matrix(S_r, S, p, p, 0);
    
    gsl_matrix * W = gsl_matrix_calloc(p,p);
    gsl_matrix * Z = gsl_matrix_calloc(p,p);
    gsl_matrix * G = gsl_matrix_alloc(p,p);
    gsl_matrix * tempG = gsl_matrix_alloc(p,p);
    gsl_matrix * Z_old = gsl_matrix_calloc(p,p);
    gsl_vector_view G_diag = gsl_matrix_diagonal(G);
    gsl_vector_view W_diag = gsl_matrix_diagonal(W);
    gsl_vector_view S_diag = gsl_matrix_diagonal(S);
    gsl_vector_view Z_diag = gsl_matrix_diagonal(Z);
    
    gsl_vector_set_all(&W_diag.vector,1.0);
    gsl_vector_div(&W_diag.vector,&S_diag.vector);
    gsl_vector_memcpy(&Z_diag.vector,&W_diag.vector);
    gsl_vector_scale(&Z_diag.vector,1.0/kappa);
    gsl_matrix_memcpy(G,S);
    gsl_vector_set_zero(&G_diag.vector);
      
    //Skip the first 0 part
    double gmax = fabs(gsl_matrix_max(G)), gmin = fabs(gsl_matrix_min(G));
    double t0 = 0.5/(gmax>gmin?gmax:gmin);
    gsl_matrix_scale(G, 2*t0);
    gsl_matrix_sub(Z, G);
    
    //Default t
    if(t_r[0] < 0)
      for (int i=0;i<nt;++i)
        t_r[i] = t0 *pow(trate,(double)i/(nt-1));
    for (int i=0;i<nt;++i)
      if(t_r[i]<=t0){
        for(int j=0; j<p; ++j){
          result_r[k*p*p+j*p+j] = gsl_matrix_get(W, j, j);
        }
        ++k;
      }
    
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    while(iter < maxiter){
        ggm_grad(S, W, G);
        gsl_matrix_transpose_memcpy(tempG,G);
        gsl_matrix_add(G,tempG);
        gsl_vector_scale(&G_diag.vector,0.5);
        gsl_matrix_scale(G, alpha);
        gsl_matrix_sub(Z, G);
        gsl_matrix_memcpy(W, Z);
        shrink_matrix_offdiag(W, 1.0);
        gsl_matrix_scale(W, kappa);
        while (k<nt & iter*alpha >= t_r[k]-t_r[0]){
          gsl_matrix_memcpy(Z_old,Z);
          gsl_matrix_memcpy(tempG,G);
          gsl_matrix_scale(tempG, (t_r[k]-t_r[0])/alpha-iter+1);
          gsl_matrix_sub(Z_old, tempG);
          shrink_matrix_offdiag(Z_old, 1.0);
          gsl_matrix_scale(Z_old, kappa);
          for(int i=0; i<p; ++i)
              for(int j=0; j<p; ++j)
                  result_r[k*p*p+j*p+i] = gsl_matrix_get(Z_old, i, j);
          ++k;
          if (*print==1){
            end = clock();
            Rprintf("%d/%d parameters computed. %f seconds used. Progress: %3.1f %%\n",k,nt,(double)(end-start)/CLOCKS_PER_SEC,(t_r[k-1]/t_r[nt-1])*100);
            //Rprintf("%f seconds used to compute gradient.\n",(double)(tm_total)/CLOCKS_PER_SEC);
          }
          if (k>=nt)
            break;
        }
        ++iter;
    }
    gsl_matrix_free(S);
    gsl_matrix_free(W);
    gsl_matrix_free(Z);
    gsl_matrix_free(G);
    gsl_matrix_free(Z_old);
    gsl_matrix_free(tempG);
}
