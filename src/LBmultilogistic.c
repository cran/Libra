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

void LB_multi_logistic_lasso(double* X_r, int*row_r, int*col_r, double*Y_r, int*category, double* kappa_r, double*alpha_r, int*iter_r, double*result_r,int*intercept)
{
    int n=*row_r, d=*col_r, r=*category, iter=0, sign=*intercept;
    double kappa = *kappa_r, alpha=*alpha_r;
    gsl_matrix *X = gsl_matrix_calloc(n, d+sign);
    gsl_matrix *Y = gsl_matrix_calloc(r, n);
    gsl_vector *temp = gsl_vector_calloc(r);
    
    read_matrix(X_r, X, n, d, 0);
    read_matrix(Y_r, Y, n, r, 1);
    
    if(sign==1){
        gsl_vector * one = gsl_vector_calloc(n);
        
        for(int i=0; i<n; ++i)
            gsl_vector_set(one, i, 1);
        gsl_matrix_set_col(X,d,one);
        ++d;
        gsl_blas_dgemv(CblasNoTrans, 1, Y, one, 0, temp);
        gsl_vector_scale(temp, 1.0/n);
        gsl_vector_log(temp);
    }
    gsl_matrix * W = gsl_matrix_calloc(r,d);
    gsl_matrix * Z = gsl_matrix_calloc(r,d);
    gsl_matrix * W_temp = gsl_matrix_calloc(r,n);
    if(sign==1){
        gsl_matrix_set_col(W,d-1,temp);
        gsl_vector_scale(temp,1.0/kappa);
        gsl_matrix_set_col(Z,d-1,temp);
    }
    while(iter < *iter_r){
        logistic_multi_grad(X, Y, W, W_temp);
        gsl_matrix_scale(W, alpha/n);
        gsl_matrix_sub(Z, W);
        gsl_matrix_memcpy(W, Z);
        gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, r, d-sign);
        shrink_matrix(&W_no_intercept.matrix, 1);
        gsl_matrix_scale(W, kappa);
        for(int i=0; i<r; ++i)
            for(int j=0; j<d; ++j)
                result_r[iter*r*d+j*r+i] = gsl_matrix_get(W, i, j);
        ++iter;
    }
}

void LB_multi_logistic_column_lasso(double* X_r, int*row_r, int*col_r, double*Y_r, int*category, double* kappa_r, double*alpha_r, int*iter_r, double*result_r,int*intercept)
{
    int n=*row_r, d=*col_r, r=*category, iter=0, sign=*intercept;
    double kappa = *kappa_r,alpha=*alpha_r;
    gsl_matrix *X = gsl_matrix_calloc(n, d+sign);
    gsl_matrix *Y = gsl_matrix_calloc(r, n);
    gsl_vector *temp = gsl_vector_calloc(r);
    
    read_matrix(X_r, X, n, d, 0);
    read_matrix(Y_r, Y, n, r, 1);
    if(sign==1){
        gsl_vector * one = gsl_vector_calloc(n);
        for(int i=0; i<n; ++i)
            gsl_vector_set(one, i, 1);
        gsl_matrix_set_col(X,d,one);
        ++d;
        gsl_blas_dgemv(CblasNoTrans, 1, Y, one, 0, temp);
        gsl_vector_scale(temp, 1.0/n);
        gsl_vector_log(temp);
    }
    gsl_matrix * W = gsl_matrix_calloc(r,d);
    gsl_matrix * Z = gsl_matrix_calloc(r,d);
    gsl_matrix * W_temp = gsl_matrix_calloc(r,n);
    if(sign==1){
        gsl_matrix_set_col(W,d-1,temp);
        gsl_vector_scale(temp,1.0/kappa);
        gsl_matrix_set_col(Z,d-1,temp);
    }
    while(iter < *iter_r){
        logistic_multi_grad(X, Y, W, W_temp);
        gsl_matrix_scale(W, alpha/n);
        gsl_matrix_sub(Z, W);
        gsl_matrix_memcpy(W, Z);
        gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, r, d-sign);
        shrink_column_matrix(&W_no_intercept.matrix);
        gsl_matrix_scale(W, kappa);
        for(int i=0; i<r; ++i)
            for(int j=0; j<d; ++j)
                result_r[iter*r*d+j*r+i] = gsl_matrix_get(W, i, j);
        ++iter;
    }
}

void LB_multi_logistic_group_lasso(double* X_r, int*row_r, int*col_r, double*Y_r, int*category, double* kappa_r, double*alpha_r, int*iter_r, double*result_r, int*group_split, int*group_split_length, int*intercept)
{
    int n=*row_r, d=*col_r, r=*category, iter=0, sign=*intercept;
    double kappa = *kappa_r, alpha=*alpha_r;
    gsl_matrix *X = gsl_matrix_calloc(n, d+sign);
    gsl_matrix *Y = gsl_matrix_calloc(r, n);
    gsl_vector *temp = gsl_vector_calloc(r);
    
    read_matrix(X_r, X, n, d, 0);
    read_matrix(Y_r, Y, n, r, 1);
    if(sign==1){
        gsl_vector * one = gsl_vector_calloc(n);
        for(int i=0; i<n; ++i)
            gsl_vector_set(one, i, 1);
        gsl_matrix_set_col(X,d,one);
        ++d;
        gsl_blas_dgemv(CblasNoTrans, 1, Y, one, 0, temp);
        gsl_vector_scale(temp, 1.0/n);
        gsl_vector_log(temp);
    }
    gsl_matrix * W = gsl_matrix_calloc(r,d);
    gsl_matrix * Z = gsl_matrix_calloc(r,d);
    gsl_matrix * W_temp = gsl_matrix_calloc(r,n);
    if(sign==1){
        gsl_matrix_set_col(W,d-1,temp);
        gsl_vector_scale(temp,1.0/kappa);
        gsl_matrix_set_col(Z,d-1,temp);
    }
    while(iter < *iter_r){
        logistic_multi_grad(X, Y, W, W_temp);
        gsl_matrix_scale(W, alpha/n);
        gsl_matrix_sub(Z, W);
        gsl_matrix_memcpy(W, Z);
        gsl_matrix_view W_no_intercept = gsl_matrix_submatrix(W, 0, 0, r, d-sign);
        shrink_group_matrix_general(&W_no_intercept.matrix, group_split, group_split_length);
        gsl_matrix_scale(W, kappa);
        for(int i=0; i<r; ++i)
            for(int j=0; j<d; ++j)
                result_r[iter*r*d+j*r+i] = gsl_matrix_get(W, i, j);
        ++iter;
    }
}
