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


void LB_logistic_lasso(double* A_r, int*row_r, int*col_r, double*y_r, double* kappa_r, double*alpha_r, int*iter_r, double*result_r, int*intercept)
{
    int row, col, m = *row_r, n = *col_r, iter=0, sign=*intercept;
    double kappa = *kappa_r, alpha = *alpha_r,temp = 0;
    gsl_matrix *A = gsl_matrix_calloc(m, n+sign);
    gsl_vector *b = gsl_vector_calloc(m);
    for(int i=0;i< m*n; ++i){
        row = i % m;
        col = floor(i/m);
        gsl_matrix_set(A, row, col, A_r[i] * y_r[row]);
    }
    for(int i=0; i<m; ++i)
        gsl_vector_set(b, i, y_r[i]);
    if(sign==1){
        for(int i=0; i<m; ++i){
            gsl_matrix_set(A, i, n, y_r[i]);
            temp += y_r[i];
        }
        temp = log((temp+m)/(m-temp));
    }
    n = (int)A->size2;
    gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *z      = gsl_vector_calloc(n);
    gsl_vector *Ax     = gsl_vector_calloc(m);
    gsl_vector *g      = gsl_vector_calloc(n);
    if(sign==1){
        gsl_vector_set(z,n-1,temp/kappa);
        gsl_vector_set(x,n-1,temp);
    }
    while(iter < *iter_r){
        
        gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A * x
        // Calculate the derivative(for both parameters and intercept)
        logistic_grad(Ax); // Ax now represents the gradient
        
        gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, g); // g = A' Ax
        gsl_vector_scale(g, alpha/m);
        gsl_vector_sub(z,g); //update z
        gsl_vector_memcpy(x, z); // use x to do the shrinkage
        
        // shrinkage step
        gsl_vector_view x_no_intercept = gsl_vector_subvector(x, 0, n-sign);
        shrink(&x_no_intercept.vector, 1); // shrink the paramters only
        gsl_vector_scale(x, kappa);
        
        // return the result
        for(int temp=0;temp<n;++temp)
            result_r[temp+iter*n] = gsl_vector_get(x, temp);
        // continue iteration
        iter++;
    }
}

void LB_logistic_group_lasso(double* A_r, int*row_r, int*col_r, double*y_r, double* kappa_r, double*alpha_r, int*iter_r, double*result_r, int*group_split, int*group_split_length, int*intercept)
{
    int row, col, m = *row_r, n = *col_r, iter=0, sign=*intercept;
    double kappa = *kappa_r, alpha = *alpha_r,temp = 0;
    gsl_matrix *A = gsl_matrix_calloc(m, n+sign);
    gsl_vector *b = gsl_vector_calloc(m);
    for(int i=0;i< m*n; ++i){
        row = i % m;
        col = floor(i/m);
        gsl_matrix_set(A, row, col, A_r[i] * y_r[row]);
    }
    for(int i=0; i<m; ++i)
        gsl_vector_set(b, i, y_r[i]);
    if(sign==1){
        for(int i=0; i<m; ++i){
            gsl_matrix_set(A, i, n, y_r[i]);
            temp += y_r[i];
        }
        temp = log((temp+m)/(m-temp));
    }
    n = (int)A->size2;
    gsl_vector *x      = gsl_vector_calloc(n);
    gsl_vector *z      = gsl_vector_calloc(n);
    gsl_vector *Ax     = gsl_vector_calloc(m);
    gsl_vector *g      = gsl_vector_calloc(n);
    if(sign==1){
        gsl_vector_set(z,n-1,temp/kappa);
        gsl_vector_set(x,n-1,temp);
    }
    while(iter < *iter_r){
        
        gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A * x
        // Calculate the derivative(for both parameters and intercept)
        logistic_grad(Ax); // Ax now represents the gradient
        
        gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, g); // g = A' Ax
        gsl_vector_scale(g, alpha/m);
        gsl_vector_sub(z,g); //update z
        gsl_vector_memcpy(x, z); // use x to do the shrinkage
        
        // shrinkage step
        gsl_vector_view x_no_intercept = gsl_vector_subvector(x, 0, n-sign);
        group_shrink_general(&x_no_intercept.vector, group_split, group_split_length); // shrink the paramters only
        gsl_vector_scale(x, kappa);
        
        // return the result
        for(int temp=0;temp<n;++temp)
            result_r[temp+iter*n] = gsl_vector_get(x, temp);
        // continue iteration
        iter++;
    }
}




