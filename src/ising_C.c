//  Created by jiechaoxiong on 15-11-11.
//  Copyright (c) 2015年 jcxiong. All rights reserved.

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

void ising_C(double* X_r, int*row_r, int*col_r, double* kappa_r, double*alpha_r,  double*result_r,int*intercept,double* t_r,int* nt_r,double* trate_r,int*print)
{
    int n=*row_r, d=*col_r, iter=0, sign=*intercept,nt=*nt_r,k=0,group_split=0,group_split_length=0;
    double kappa = *kappa_r, alpha=*alpha_r, trate=*trate_r;
    time_t start = clock(),end;
    gsl_matrix *X = gsl_matrix_alloc(n, d+sign);
    gsl_vector *temp = gsl_vector_alloc(d+sign);
    gsl_vector_view tmp = gsl_vector_subvector(temp,0,d);
    
    read_matrix(X_r, X, n, d, 0);
    if(sign==1){
        gsl_vector* one = gsl_vector_alloc(n);
        gsl_vector_set_all(one,1);
        gsl_matrix_set_col(X,d,one);
        gsl_blas_dgemv(CblasTrans, 1, X, one, 0, temp);
        for(int i=0; i<d; ++i)
          gsl_vector_set(temp,i,log(2*n/(n-gsl_vector_get(temp,i))-1));
        gsl_vector_free(one);
    }
    gsl_matrix * W = gsl_matrix_calloc(d,d+sign);
    gsl_matrix * Z = gsl_matrix_calloc(d,d+sign);
    gsl_matrix * G = gsl_matrix_alloc(d,d+sign);
    gsl_matrix * Z_old = gsl_matrix_calloc(d,d+sign);
    gsl_matrix * W_temp = gsl_matrix_alloc(d,n);
    gsl_matrix * tempG = gsl_matrix_alloc(d,d+sign);
    gsl_matrix_view G_no_inter = gsl_matrix_submatrix(G,0,0,d,d);
    gsl_matrix_view tempG_no_inter = gsl_matrix_submatrix(tempG,0,0,d,d);
    gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, d, d);
    gsl_matrix_view Z_no_intercept = gsl_matrix_submatrix(Z_old, 0, 0, d, d);
    
    if(sign==1){
        gsl_matrix_set_col(W,d,&tmp.vector);
        gsl_vector_scale(&tmp.vector,1.0/kappa);
        gsl_matrix_set_col(Z,d,&tmp.vector);
    }
    
    //Skip the first 0 part
    ising_grad(X, W,W_temp, G); //
    gsl_matrix_transpose_memcpy(&tempG_no_inter.matrix,&G_no_inter.matrix);
    gsl_matrix_add(&G_no_inter.matrix,&tempG_no_inter.matrix);
    double gmax = fabs(gsl_matrix_max(&G_no_inter.matrix)), gmin = fabs(gsl_matrix_min(&G_no_inter.matrix));
    double t0 = n/(gmax>gmin?gmax:gmin);
    gsl_matrix_scale(G, t0/n);
    gsl_matrix_sub(Z, G);
    
    //Default t
    if(t_r[0] < 0)
      for (int i=0;i<nt;++i)
        t_r[i] = t0 *pow(trate,(double)i/(nt-1));
    for (int i=0;i<nt;++i)
      if(t_r[i]<=t0){
        if (sign==1){
          for(int i=0; i<d; ++i)
            result_r[k*d*(d+sign)+d*d+i] = gsl_matrix_get(W, i, d);
        }
        ++k;
      }
    
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    while(iter < maxiter){
        ising_grad(X, W,W_temp, G);
        gsl_matrix_transpose_memcpy(&tempG_no_inter.matrix,&G_no_inter.matrix);
        gsl_matrix_add(&G_no_inter.matrix,&tempG_no_inter.matrix);
        gsl_matrix_scale(G, alpha/n);
        gsl_matrix_sub(Z, G);
        gsl_matrix_memcpy(W, Z);
        general_shrink_matrix(&W_no_intercept.matrix, &group_split, &group_split_length);
        gsl_matrix_scale(W, kappa);
        while (k<nt && iter*alpha >= t_r[k]-t_r[0]){
          gsl_matrix_memcpy(Z_old,Z);
          gsl_matrix_memcpy(tempG,G);
          gsl_matrix_scale(tempG, (t_r[k]-t_r[0])/alpha-iter+1);
          gsl_matrix_sub(Z_old, tempG);
          general_shrink_matrix(&Z_no_intercept.matrix, &group_split, &group_split_length); // shrink the paramters only
          gsl_matrix_scale(Z_old, kappa);
          for(int i=0; i<d; ++i)
              for(int j=0; j<d+sign; ++j)
                  result_r[k*d*(d+sign)+j*d+i] = gsl_matrix_get(Z_old, i, j);
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
    gsl_matrix_free(X);
    gsl_matrix_free(W);
    gsl_matrix_free(Z);
    gsl_matrix_free(G);
    gsl_matrix_free(Z_old);
    gsl_vector_free(temp);
    gsl_matrix_free(tempG);
    gsl_matrix_free(W_temp);
}
