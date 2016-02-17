//
//  Lasso(LBparallel).c
//  Lasso(LBparallel)
//
//  Created by frankruan on 13-10-4.
//  Copyright (c) 2013年 frankruan. All rights reserved.
//  Modified by jiechaoxiong on 15-10-19

#ifndef LBHeader_h
#define LBHeader_h

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

void shrink_vector(gsl_vector *v, double sigma);
void group_shrink_vector(gsl_vector* v, int *group_split, int*group_split_length);
void general_shrink_vector(gsl_vector *v,int *group_split, int*group_split_length);
void lasso_grad(gsl_matrix* A,gsl_vector* b,gsl_vector*x,gsl_vector* Ax,gsl_vector*w);
void logistic_grad(gsl_matrix* A,gsl_vector* b,gsl_vector*x,gsl_vector* Ax,gsl_vector*w);
void read_matrix(double* X, gsl_matrix* Y, int n, int p, int trans);
void logistic_multi_grad(gsl_matrix* X, gsl_matrix* Y, gsl_matrix* W, gsl_matrix* W_temp,gsl_matrix* G);
void gsl_matrix_col_scale(gsl_matrix *X);
void shrink_matrix(gsl_matrix *v, double sigma);
void shrink_matrix_offdiag(gsl_matrix *v, double sigma);
void column_shrink_matrix(gsl_matrix *v);
void group_shrink_matrix(gsl_matrix *v,int *group_split, int*group_split_length);
void block_shrink_matrix(gsl_matrix *v,int *group_split, int*group_split_length);
void general_shrink_matrix(gsl_matrix *v,int *group_split, int*group_split_length);
void gsl_matrix_exp(gsl_matrix* X);
void gsl_vector_log(gsl_vector* v);
double gsl_vector_sum(const gsl_vector* v);
void gsl_vector_inv(gsl_vector* v);
void gsl_matrix_col_sum(const gsl_matrix *X,gsl_vector *v);
void gsl_matrix_get_diag(const gsl_matrix *X,gsl_vector *v);
void gsl_matrix_sub_diag(gsl_matrix *X,const gsl_vector *v);
double gsl_matrix_Fnorm(gsl_matrix* X);
void gsl_matrix_col_scale_v(gsl_matrix *X,const gsl_vector *v);
void ising_grad(gsl_matrix* X,gsl_matrix*W,gsl_matrix*W_temp,gsl_matrix*G);
void potts_grad(gsl_matrix* X,gsl_matrix* XT,gsl_matrix*W,gsl_matrix*W_temp,gsl_matrix*G,int *group_split, int*group_split_length);
void ggm_grad(gsl_matrix* S,gsl_matrix*W,gsl_matrix*G);
#endif


