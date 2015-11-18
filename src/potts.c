//  Created by jiechaoxiong on 15-11-11.
//  Copyright (c) 2015年 jcxiong. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include "LBheader.h"

void potts_entry(double* X_r, int*row_r, int*col_r, double* kappa_r, double*alpha_r,  double*result_r,int*group_split, int*group_split_length,int*intercept,double* t_r,int* nt_r,double* trate_r)
{
    int n=*row_r, d=*col_r, iter=0, sign=*intercept,nt=*nt_r,k=0;
    double kappa = *kappa_r, alpha=*alpha_r, trate=*trate_r,c_old,c_new;
    gsl_matrix *X = gsl_matrix_alloc(n, d+sign);
    gsl_vector *temp = gsl_vector_alloc(d+sign);
    gsl_vector_view tmp = gsl_vector_subvector(temp,0,d);
    
    read_matrix(X_r, X, n, d, 0);
    if(sign==1){
        gsl_vector* one = gsl_vector_alloc(n);
        gsl_vector_set_all(one,1);
        gsl_matrix_set_col(X,d,one);
        gsl_blas_dgemv(CblasTrans, 1, X, one, 0, temp);
        gsl_vector_scale(temp, 1.0/n);
        gsl_vector_log(temp);
    }
    gsl_matrix * W = gsl_matrix_calloc(d,d+sign);
    gsl_matrix * Z = gsl_matrix_calloc(d,d+sign);
    gsl_matrix * G = gsl_matrix_alloc(d,d+sign);
    gsl_matrix * W_old = gsl_matrix_calloc(d,d+sign);
    gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, d, d);
    
    if(sign==1){
        gsl_matrix_set_col(W,d,&tmp.vector);
        gsl_vector_scale(&tmp.vector,1.0/kappa);
        gsl_matrix_set_col(Z,d,&tmp.vector);
    }
    
    //Skip the first 0 part
    potts_grad(X, W, G, group_split, group_split_length); // 
    gsl_matrix_view G_no_intercept = gsl_matrix_submatrix(G, 0, 0,d,d);
    double gmax = fabs(gsl_matrix_max(&G_no_intercept.matrix)), gmin = fabs(gsl_matrix_min(&G_no_intercept.matrix));
    double t0 = n/(gmax>gmin?gmax:gmin);
    gsl_matrix_scale(G, t0/n);
    gsl_matrix_sub(Z, G);
    
    //Default t
    if(t_r[0] < 0)
      for (int i=0;i<nt;++i)
        t_r[i] = t0 *pow(trate,(double)i/(nt-1));
    for (int i=0;i<nt;++i)
      if(t_r[i]<=t0) ++k;
    
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    while(iter < maxiter){
        potts_grad(X, W, G, group_split, group_split_length);
        gsl_matrix_scale(G, alpha/n);
        gsl_matrix_sub(Z, G);
        gsl_matrix_memcpy(W, Z);
        shrink_matrix(&W_no_intercept.matrix, 1);
        gsl_matrix_scale(W, kappa);
        while (k<nt & iter*alpha >= t_r[k]-t_r[0]){
          c_old = iter-(t_r[k]-t_r[0])/alpha;
          c_new = (t_r[k]-t_r[0])/alpha-iter+1;
          for(int i=0; i<d; ++i)
              for(int j=0; j<d+sign; ++j)
                  result_r[k*d*(d+sign)+j*d+i] = gsl_matrix_get(W, i, j)*c_new +gsl_matrix_get(W_old, i, j)*c_old;
          ++k;
        }
        if (k>=nt)
          break;
        gsl_matrix_memcpy(W_old,W);
        ++iter;
    }
}
void potts_block(double* X_r, int*row_r, int*col_r, double* kappa_r, double*alpha_r,  double*result_r,int*group_split, int*group_split_length,int*intercept,double* t_r,int* nt_r,double* trate_r)
{
  int n=*row_r, d=*col_r, iter=0, sign=*intercept,nt=*nt_r,k=0;
  double kappa = *kappa_r, alpha=*alpha_r, trate=*trate_r,c_old,c_new;
  gsl_matrix *X = gsl_matrix_alloc(n, d+sign);
  gsl_vector *temp = gsl_vector_alloc(d+sign);
  gsl_vector_view tmp = gsl_vector_subvector(temp,0,d);
  
  read_matrix(X_r, X, n, d, 0);
  if(sign==1){
    gsl_vector* one = gsl_vector_alloc(n);
    gsl_vector_set_all(one,1);
    gsl_matrix_set_col(X,d,one);
    gsl_blas_dgemv(CblasTrans, 1, X, one, 0, temp);
    gsl_vector_scale(temp, 1.0/n);
    gsl_vector_log(temp);
  }
  gsl_matrix * W = gsl_matrix_calloc(d,d+sign);
  gsl_matrix * Z = gsl_matrix_calloc(d,d+sign);
  gsl_matrix * G = gsl_matrix_alloc(d,d+sign);
  gsl_matrix * W_old = gsl_matrix_calloc(d,d+sign);
  gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, d, d);
  
  if(sign==1){
    gsl_matrix_set_col(W,d,&tmp.vector);
    gsl_vector_scale(&tmp.vector,1.0/kappa);
    gsl_matrix_set_col(Z,d,&tmp.vector);
  }
  
  //Skip the first 0 part
  potts_grad(X, W, G, group_split, group_split_length); // 
  gsl_matrix_view G_no_intercept = gsl_matrix_submatrix(G, 0, 0,d,d);
  gsl_vector *gp_norm = gsl_vector_alloc((*group_split_length-1)*(*group_split_length-1));
  for(int i=0; i<((*group_split_length)-1); ++i){
    for(int j=0; j<((*group_split_length)-1); ++j){
      gsl_matrix_view group_i = gsl_matrix_submatrix(G, group_split[i],group_split[j], group_split[i+1]-group_split[i], group_split[j+1]-group_split[j]);
      gsl_vector_set(gp_norm,i+j*(*group_split_length-1),gsl_matrix_Fnorm(&group_i.matrix));
    }
  }
  int q = gsl_blas_idamax(gp_norm);
  double t0 = n/fabs(gsl_vector_get(gp_norm,q));
  gsl_matrix_scale(G, t0/n);
  gsl_matrix_sub(Z, G);
  
  //Default t
  if(t_r[0] < 0)
    for (int i=0;i<nt;++i)
      t_r[i] = t0 *pow(trate,(double)i/(nt-1));
  for (int i=0;i<nt;++i)
    if(t_r[i]<=t0) ++k;
    
    double maxiter = (t_r[nt-1]-t_r[0])/alpha+1;
    while(iter < maxiter){
      potts_grad(X, W, G, group_split, group_split_length);
      gsl_matrix_scale(G, alpha/n);
      gsl_matrix_sub(Z, G);
      gsl_matrix_memcpy(W, Z);
      shrink_block_matrix_general(&W_no_intercept.matrix, group_split, group_split_length);
      gsl_matrix_scale(W, kappa);
      while (k<nt & iter*alpha >= t_r[k]-t_r[0]){
        c_old = iter-(t_r[k]-t_r[0])/alpha;
        c_new = (t_r[k]-t_r[0])/alpha-iter+1;
        for(int i=0; i<d; ++i)
          for(int j=0; j<d+sign; ++j)
            result_r[k*d*(d+sign)+j*d+i] = gsl_matrix_get(W, i, j)*c_new +gsl_matrix_get(W_old, i, j)*c_old;
        ++k;
      }
      if (k>=nt)
        break;
      gsl_matrix_memcpy(W_old,W);
      ++iter;
    }
}
