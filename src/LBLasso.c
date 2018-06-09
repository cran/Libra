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

void LB_lasso(double* A_r, int*row_r, int*col_r, double*y_r, double* kappa_r, double*alpha_r,double*alpha0_rate_r, double*result_r, int*group_split, int*group_split_length, int*intercept,double* t_r,int* nt_r,double* trate_r,int*print)
{
    int m = *row_r, n = *col_r, iter=0, sign=*intercept,nt=*nt_r, k=0;
    double kappa = *kappa_r, alpha = *alpha_r,alpha0_rate = *alpha0_rate_r,temp = 0, trate=*trate_r;
    time_t start = clock(),end;
    gsl_matrix *A = gsl_matrix_alloc(m, n+sign);
    gsl_vector *b = gsl_vector_alloc(m);
    
    read_matrix(A_r, A, m, n, 0);
    for(int i=0; i<m; ++i)
        gsl_vector_set(b, i, y_r[i]);
    if(sign==1){
        gsl_vector * one = gsl_vector_alloc(m);
        gsl_vector_set_all(one,1);
        
        gsl_matrix_set_col(A,n,one);
        ++n;
        gsl_blas_ddot(b,one,&temp);
        temp = temp/m;
        gsl_vector_free(one);
    }
    gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *z      = gsl_vector_calloc(n);
    gsl_vector *Ax     = gsl_vector_alloc(m);
    gsl_vector *g      = gsl_vector_alloc(n);
    gsl_vector *z_old  = gsl_vector_calloc(n);
    gsl_vector *g_old  = gsl_vector_calloc(n);
    gsl_vector_view x_no_intercept = gsl_vector_subvector(x, 0, n-sign);
    gsl_vector_view z_no_intercept = gsl_vector_subvector(z_old, 0, n-sign);
    
    // Initialize
    if(sign==1){
        gsl_vector_set(z,n-1,temp/kappa);
        gsl_vector_set(x,n-1,temp);
    }
    
    //Skip the first 0 part
    double t0;
    lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; w = A'*AX
    if (*group_split_length==0){
      gsl_vector_view g_no_intercept = gsl_vector_subvector(g, 0, n-sign);
      int q = gsl_blas_idamax(&g_no_intercept.vector);
      t0 = m/fabs(gsl_vector_get(&g_no_intercept.vector,q));
    }else{
      gsl_vector *gp_norm = gsl_vector_alloc((*group_split_length)-1);
      for(int i=0; i<((*group_split_length)-1); ++i){
        gsl_vector_view group_i = gsl_vector_subvector(g, group_split[i], (group_split[i+1]-group_split[i]));
        gsl_vector_set(gp_norm,i,gsl_blas_dnrm2(&group_i.vector));
      }
      int q = gsl_blas_idamax(gp_norm);
      t0 = m/fabs(gsl_vector_get(gp_norm,q));
      gsl_vector_free(gp_norm);
    }
    gsl_vector_scale(g, t0/m);
    gsl_vector_sub(z, g);
    //Default t
    if(t_r[0] < 0)
      for (int temp=0;temp<nt;++temp)
        t_r[temp] = t0 *pow(trate,(double)temp/(nt-1));
    for (int temp=0;temp<nt;++temp)
      if(t_r[temp]<=t0){
        if (sign==1)
          result_r[k*n+n-1] = gsl_vector_get(x, n-sign);
        ++k;
      }
      
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
      
    while(iter < maxiter){
        lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; w = A'*Ax
        gsl_vector_scale(g, alpha/m);
        if(sign==1){
          gsl_vector_set(g,n-1,gsl_vector_get(g,n-1)*alpha0_rate);
        }
        gsl_vector_sub(z, g); //update z
        gsl_vector_memcpy(x, z); // use x to do the shrinkage
        
        // shrinkage step
        general_shrink_vector(&x_no_intercept.vector, group_split, group_split_length); // shrink the paramters only
        gsl_vector_scale(x, kappa);
        
        // return the result
        while (k<nt && iter*alpha >= t_r[k]-t_r[0]){
          gsl_vector_memcpy(z_old,z);
          gsl_vector_memcpy(g_old,g);
          gsl_vector_scale(g_old, (t_r[k]-t_r[0])/alpha-iter);
          gsl_vector_sub(z_old, g_old);
          general_shrink_vector(&z_no_intercept.vector, group_split, group_split_length); // shrink the paramters only
          gsl_vector_scale(z_old, kappa);
          for(int temp=0;temp<n;++temp)
            result_r[temp+k*n] = gsl_vector_get(z_old, temp);
          ++k;
          if (*print==1){
            end = clock();
            Rprintf("%d/%d parameters computed. %f seconds used. Progress: %3.1f %%\n",k,nt,(double)(end-start)/CLOCKS_PER_SEC,(t_r[k-1]/t_r[nt-1])*100);
          }
          if (k>=nt)
            break;
        }
        // continue iteration
        iter++;
    }
    gsl_matrix_free(A);
    gsl_vector_free(b);
    gsl_vector_free(x);
    gsl_vector_free(z);
    gsl_vector_free(g);
    gsl_vector_free(Ax);
    gsl_vector_free(z_old);
    gsl_vector_free(g_old);
}
