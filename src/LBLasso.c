//  Created by frankruan on 13-10-4.
//  Copyright (c) 2013年 frankruan. All rights reserved.
//  Modified by jiechaoxiong on 15-10-19

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include "LBheader.h"

void LB_lasso(double* A_r, int*row_r, int*col_r, double*y_r, double* kappa_r, double*alpha_r,double*alpha0_rate_r, double*result_r, int*intercept,double* t_r,int* nt_r,double* trate_r)
{
    int m = *row_r, n = *col_r, iter=0, sign=*intercept,nt=*nt_r, k=0;
    double kappa = *kappa_r, alpha = *alpha_r,alpha0_rate = *alpha0_rate_r, temp, trate=*trate_r,c_old,c_new;
    gsl_matrix *A = gsl_matrix_alloc(m, n+sign);
    gsl_vector *b = gsl_vector_alloc(m);
    
    //Read data and initialize
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
    }
    gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *g      = gsl_vector_alloc(n);
    gsl_vector *Ax     = gsl_vector_alloc(m);
    gsl_vector *z      = gsl_vector_calloc(n);
    gsl_vector *x_old  = gsl_vector_calloc(n);
    gsl_vector_view x_no_intercept = gsl_vector_subvector(x, 0, n-sign);
    if(sign==1){
        gsl_vector_set(z,n-1,temp/kappa);
        gsl_vector_set(x,n-1,temp);
    }
    
    //Skip the first 0 part
    lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; w = A'*AX
    gsl_vector_view g_no_intercept = gsl_vector_subvector(g, 0, n-sign);
    int q = gsl_blas_idamax(&g_no_intercept.vector);
    double t0 = m/fabs(gsl_vector_get(&g_no_intercept.vector,q));
    gsl_vector_scale(g, t0/m);
    gsl_vector_sub(z, g);
    
    //Default t
    if(t_r[0] < 0)
      for (int temp=0;temp<nt;++temp)
        t_r[temp] = t0 *pow(trate,(double)temp/(nt-1));
    for (int temp=0;temp<nt;++temp)
      if(t_r[temp]<=t0) ++k;
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    
    //Iteration
    while(iter < maxiter){
        lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; g = A'*Ax
        gsl_vector_scale(g, alpha/m);
        gsl_vector_sub(z, g); //update z
        gsl_vector_memcpy(x, z); // use x to do the shrinkage
        
        // shrinkage step
        shrink(&x_no_intercept.vector, 1);
        gsl_vector_scale(x, kappa);
        if(sign==1){
          gsl_vector_set(x,n-1,gsl_vector_get(x,n-1)*alpha0_rate);
        }
        
        // return the result
        while (k<nt & iter*alpha >= t_r[k]-t_r[0]){
          c_old = iter-(t_r[k]-t_r[0])/alpha;
          c_new = (t_r[k]-t_r[0])/alpha-iter+1;
          for(int temp=0;temp<n;++temp)
            result_r[temp+k*n] = gsl_vector_get(x, temp)*c_new+gsl_vector_get(x_old, temp)*c_old;
          ++k;
        }
        if (k>=nt)
          break;
        gsl_vector_memcpy(x_old,x);
        // continue iteration
        iter++;
    }
}

void LB_group_lasso(double* A_r, int*row_r, int*col_r, double*y_r, double* kappa_r, double*alpha_r,double*alpha0_rate_r, double*result_r, int*group_split, int*group_split_length, int*intercept,double* t_r,int* nt_r,double* trate_r)
{
    int m = *row_r, n = *col_r, iter=0, sign=*intercept,nt=*nt_r, k=0;
    double kappa = *kappa_r, alpha = *alpha_r,alpha0_rate = *alpha0_rate_r,temp = 0, trate=*trate_r,c_old,c_new;
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
    }
    gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *z      = gsl_vector_calloc(n);
    gsl_vector *Ax     = gsl_vector_alloc(m);
    gsl_vector *g      = gsl_vector_alloc(n);
    gsl_vector *x_old  = gsl_vector_calloc(n);
    gsl_vector_view x_no_intercept = gsl_vector_subvector(x, 0, n-sign);
    if(sign==1){
        gsl_vector_set(z,n-1,temp/kappa);
        gsl_vector_set(x,n-1,temp);
    }
    
    // Initialize
    lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; w = A'*AX
    gsl_vector *gp_norm = gsl_vector_alloc((*group_split_length)-1);
    for(int i=0; i<((*group_split_length)-1); ++i){
      gsl_vector_view group_i = gsl_vector_subvector(g, group_split[i], (group_split[i+1]-group_split[i]));
      gsl_vector_set(gp_norm,i,gsl_blas_dnrm2(&group_i.vector));
    }
    int q = gsl_blas_idamax(gp_norm);
    double t0 = m/fabs(gsl_vector_get(gp_norm,q));
    gsl_vector_scale(g, t0/m);
    gsl_vector_sub(z, g);
    
    //Default t
    if(t_r[0] < 0)
      for (int temp=0;temp<nt;++temp)
        t_r[temp] = t0 *pow(trate,(double)temp/(nt-1));
    for (int temp=0;temp<nt;++temp)
      if(t_r[temp]<=t0) ++k;
      
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    while(iter < maxiter){
        lasso_grad(A,b,x,Ax,g); // Ax = A*x-b; w = A'*Ax
        gsl_vector_scale(g, alpha/m);
        gsl_vector_sub(z, g); //update z
        gsl_vector_memcpy(x, z); // use x to do the shrinkage
        
        // shrinkage step
        group_shrink_general(&x_no_intercept.vector, group_split, group_split_length); // shrink the paramters only
        gsl_vector_scale(x, kappa);
        if(sign==1){
          gsl_vector_set(x,n-1,gsl_vector_get(x,n-1)*alpha0_rate);
        }
        
        // return the result
        while (k<nt & iter*alpha >= t_r[k]-t_r[0]){
          c_old = iter-(t_r[k]-t_r[0])/alpha;
          c_new = (t_r[k]-t_r[0])/alpha-iter+1;
          for(int temp=0;temp<n;++temp)
            result_r[temp+k*n] = gsl_vector_get(x, temp)*c_new+gsl_vector_get(x_old, temp)*c_old;
          ++k;
        }
        if (k>=nt)
          break;
        gsl_vector_memcpy(x_old,x);
        // continue iteration
        iter++;
    }
}
