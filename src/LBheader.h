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

void shrink(gsl_vector *v, double sigma);
void group_shrink_general(gsl_vector* v, int *group_split, int*group_split_length);
void logistic_grad(gsl_vector* v);
void dummy_generation(double* Y_r, gsl_matrix* Y_dummy, int* n);
void logistic_multi_grad(gsl_matrix* X, gsl_matrix* Y, gsl_matrix* W, gsl_matrix* W_temp);
void gsl_matrix_col_scale(gsl_matrix *X);
void shrink_matrix(gsl_matrix *v, double sigma);
void shrink_group_matrix(gsl_matrix *v);
void shrink_block_matrix_general(gsl_matrix *v,int *group_split, int*group_split_length);
void gsl_matrix_exp(gsl_matrix* X);
void gsl_vector_log(gsl_vector* v);
double gsl_matrix_Fnorm(gsl_matrix* X);

#endif


