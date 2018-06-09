//  Created by frankruan on 13-10-4.
//  Copyright (c) 2013年 frankruan. All rights reserved.
//  Modified by jiechaoxiong on 15-10-19

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

void LB_multi_logistic(double* X_r, int*row_r, int*col_r, double*Y_r, int*category, double* kappa_r, double*alpha_r,double*alpha0_rate_r,double*result_r, int*group_split, int*group_split_length, int*intercept,double* t_r,int* nt_r,double* trate_r,int*print)
{
    int n=*row_r, d=*col_r, r=*category, iter=0, sign=*intercept,nt=*nt_r, k=0;
    double kappa = *kappa_r, alpha=*alpha_r,alpha0_rate = *alpha0_rate_r, trate=*trate_r;
    time_t start = clock(),end;
    gsl_matrix *X = gsl_matrix_alloc(n, d+sign);
    gsl_matrix *Y = gsl_matrix_alloc(r, n);
    gsl_vector *temp = gsl_vector_alloc(r);
    
    read_matrix(X_r, X, n, d, 0);
    read_matrix(Y_r, Y, n, r, 1);
    if(sign==1){
        gsl_vector * one = gsl_vector_alloc(n);
        gsl_vector_set_all(one,1);
        
        gsl_matrix_set_col(X,d,one);
        ++d;
        gsl_blas_dgemv(CblasNoTrans, 1, Y, one, 0, temp);
        gsl_vector_scale(temp, 1.0/n);
        gsl_vector_log(temp);
        gsl_vector_free(one);
    }
    gsl_matrix * W = gsl_matrix_calloc(r,d);
    gsl_matrix * Z = gsl_matrix_calloc(r,d);
    gsl_matrix * G = gsl_matrix_alloc(r,d);
    gsl_matrix * W_temp = gsl_matrix_alloc(r,n);
    gsl_matrix * Z_old = gsl_matrix_calloc(r,d);
    gsl_matrix * G_old = gsl_matrix_calloc(r,d);
    gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, r, d-sign);
    gsl_matrix_view Z_no_intercept = gsl_matrix_submatrix(Z_old, 0, 0, r, d-sign);
    // Initialize
    if(sign==1){
        gsl_matrix_set_col(W,d-1,temp);
        gsl_vector_scale(temp,1.0/kappa);
        gsl_matrix_set_col(Z,d-1,temp);
    }
    
    //Skip the first 0 part
    double t0;
    logistic_multi_grad(X,Y,W,W_temp,G); //
    if (*group_split_length==0){
      gsl_matrix_view G_no_intercept = gsl_matrix_submatrix(G, 0, 0,r,d-sign);
      double gmax = fabs(gsl_matrix_max(&G_no_intercept.matrix)), gmin = fabs(gsl_matrix_min(&G_no_intercept.matrix));
      t0 = n/(gmax>gmin?gmax:gmin);
    }else if(*group_split_length==1){
      gsl_vector *gp_norm = gsl_vector_alloc(d-sign);
      for(int i=0; i<(d-sign); ++i){
        gsl_vector_view group_i = gsl_matrix_column(G, i);
        gsl_vector_set(gp_norm,i,gsl_blas_dnrm2(&group_i.vector));
      }
      int q = gsl_blas_idamax(gp_norm);
      t0 = n/fabs(gsl_vector_get(gp_norm,q));
      gsl_vector_free(gp_norm);
    }else{
      gsl_vector *gp_norm = gsl_vector_alloc((*group_split_length)-1);
      for(int i=0; i<((*group_split_length)-1); ++i){
        gsl_matrix_view group_i = gsl_matrix_submatrix(G, 0, group_split[i], r, group_split[i+1]-group_split[i]);
        gsl_vector_set(gp_norm,i,gsl_matrix_Fnorm(&group_i.matrix));
      }
      int q = gsl_blas_idamax(gp_norm);
      t0 = n/fabs(gsl_vector_get(gp_norm,q));
      gsl_vector_free(gp_norm);
    }
    gsl_matrix_scale(G, t0/n);
    gsl_matrix_sub(Z, G);
    
    //Default t
    if(t_r[0] < 0)
      for (int temp=0;temp<nt;++temp)
        t_r[temp] = t0 *pow(trate,(double)temp/(nt-1));
    for (int temp=0;temp<nt;++temp)
      if(t_r[temp]<=t0){
        if (sign==1)
          for(int i=0; i<r; ++i)
            result_r[k*r*d+(d-1)*r+i] = gsl_matrix_get(W, i, d-1);
        ++k;
      }
      
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    
    while(iter < maxiter){
        logistic_multi_grad(X, Y, W, W_temp,G);
        gsl_matrix_scale(G, alpha/n);
        if(sign==1){
          gsl_matrix_get_col(temp,G,d-1);
          gsl_vector_scale(temp,alpha0_rate);
          gsl_matrix_set_col(G,d-1,temp);
        }
        gsl_matrix_sub(Z, G);
        gsl_matrix_memcpy(W, Z);
        
        general_shrink_matrix(&W_no_intercept.matrix, group_split, group_split_length);
        gsl_matrix_scale(W, kappa);
        
        while (k<nt && iter*alpha >= t_r[k]-t_r[0]){
          gsl_matrix_memcpy(Z_old,Z);
          gsl_matrix_memcpy(G_old,G);
          gsl_matrix_scale(G_old, (t_r[k]-t_r[0])/alpha-iter);
          gsl_matrix_sub(Z_old, G_old);
          general_shrink_matrix(&Z_no_intercept.matrix, group_split, group_split_length); // shrink the paramters only
          gsl_matrix_scale(Z_old, kappa);
          for(int i=0; i<r; ++i)
            for(int j=0; j<d; ++j)
              result_r[k*r*d+j*r+i] = gsl_matrix_get(Z, i, j);
          ++k;
          if (*print==1){
            end = clock();
            Rprintf("%d/%d parameters computed. %f seconds used. Progress: %3.1f %%\n",k,nt,(double)(end-start)/CLOCKS_PER_SEC,(t_r[k-1]/t_r[nt-1])*100);
          }
          if (k>=nt)
            break;
        }
        ++iter;
    }
    gsl_matrix_free(X);
    gsl_matrix_free(Y);
    gsl_matrix_free(W);
    gsl_matrix_free(Z);
    gsl_matrix_free(G);
    gsl_matrix_free(W_temp);
    gsl_matrix_free(Z_old);
    gsl_matrix_free(G_old);
    gsl_vector_free(temp);
}
