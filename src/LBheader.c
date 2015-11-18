//  Created by frankruan on 13-10-4.
//  Copyright (c) 2013年 frankruan. All rights reserved.
//  Modified by jiechaoxiong on 15-10-19


#include "LBheader.h"
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

void read_matrix(double* X, gsl_matrix* Y, int n, int p, int trans){
  int row, col;
  for(int i=0;i< n*p; ++i){
    row = i % n;
    col = floor(i/n);
    if(trans==1) {gsl_matrix_set(Y, col,row, X[i]);}
    else {gsl_matrix_set(Y, row,col, X[i]);}
  }
}

void lasso_grad(gsl_matrix* A,gsl_vector* b,gsl_vector*x,gsl_vector* Ax,gsl_vector*g){
  gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax); // Ax = A * x
  gsl_vector_sub(Ax,b); // Ax = A*x-b;
  gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, g); //g = A'*(A*x-b)
}

void logistic_grad(gsl_matrix* A,gsl_vector* b,gsl_vector*x,gsl_vector* Ax,gsl_vector*g){
    gsl_blas_dgemv(CblasNoTrans, 1, A, x, 0, Ax);
    int m = (int)Ax->size;
    for(int i=0; i<m; ++i)
        gsl_vector_set(Ax, i, -gsl_vector_get(b, i)/(1+exp(gsl_vector_get(b, i)*gsl_vector_get(Ax, i))));
    gsl_blas_dgemv(CblasTrans, 1, A, Ax, 0, g);
}

void logistic_multi_grad(gsl_matrix* X, gsl_matrix* Y, gsl_matrix* W, gsl_matrix* W_temp,gsl_matrix*G){
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, W, X, 0, W_temp);
    gsl_matrix_exp(W_temp);
    gsl_matrix_col_scale(W_temp);
    gsl_matrix_sub(W_temp,Y);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, W_temp, X, 0, G);// G is gradient
}

void ising_grad(gsl_matrix* X,gsl_matrix*W,gsl_matrix*G){
  int n = X->size1, d = W->size1;
  gsl_vector* temp = gsl_vector_alloc(n);
  for(int i=0;i<d;++i){
    gsl_vector_view b = gsl_matrix_column(X,i);
    gsl_vector_view x = gsl_matrix_row(W,i);
    gsl_vector_view g = gsl_matrix_row(G,i);
    logistic_grad(X,&b.vector,&x.vector,temp,&g.vector);
    gsl_vector_set(&g.vector,i,0);
  }
  gsl_matrix_view G_no_inter = gsl_matrix_submatrix(G,0,0,d,d);
  gsl_matrix* tempG = gsl_matrix_alloc(d,d);
  gsl_matrix_transpose_memcpy(tempG,&G_no_inter.matrix);
  gsl_matrix_add(&G_no_inter.matrix,tempG);
  gsl_vector_free(temp);
  gsl_matrix_free(tempG);
}

void potts_grad(gsl_matrix* X,gsl_matrix*W,gsl_matrix*G,int *group_split, int*group_split_length){
  int n = X->size1, d = W->size1,p = W->size2, start, length;
  for(int i=0;i<(*group_split_length)-1;++i){
    length = (group_split[i+1]-group_split[i]);
    start = group_split[i];
    gsl_matrix_view b = gsl_matrix_submatrix(X,0,start,n,length);
    gsl_matrix* Y = gsl_matrix_alloc(length,n);
    gsl_matrix* W_temp = gsl_matrix_alloc(length,n);
    gsl_matrix_transpose_memcpy(Y,&b.matrix);
    gsl_matrix_view x = gsl_matrix_submatrix(W,start,0,length,p);
    gsl_matrix_view g = gsl_matrix_submatrix(G,start,0,length,p);
    logistic_multi_grad(X,Y,&x.matrix,W_temp,&g.matrix);
    gsl_matrix_view gdiag = gsl_matrix_submatrix(G,start,start,length,length);
    gsl_matrix_set_zero(&gdiag.matrix);
    gsl_matrix_free(Y);
    gsl_matrix_free(W_temp);
  }
  gsl_matrix_view G_no_inter = gsl_matrix_submatrix(G,0,0,d,d);
  gsl_matrix* tempG = gsl_matrix_alloc(d,d);
  gsl_matrix_transpose_memcpy(tempG,&G_no_inter.matrix);
  gsl_matrix_add(&G_no_inter.matrix,tempG);
  gsl_matrix_free(tempG);
}

void shrink(gsl_vector *v, double sigma) {
  double vi;
  for (int i = 0; i < v->size; ++i) {
    vi = gsl_vector_get(v, i);
    if (vi > sigma)       { gsl_vector_set(v, i, vi-sigma); }
    else if (vi < -sigma) { gsl_vector_set(v, i, vi+sigma); }
    else              { gsl_vector_set(v, i, 0); }
  }
}

void group_shrink_general(gsl_vector* v, int *group_split, int*group_split_length){
  for(int i=0; i<((*group_split_length)-1); ++i){
    gsl_vector_view group_i = gsl_vector_subvector(v, group_split[i], (group_split[i+1]-group_split[i]));
    double group_i_norm = gsl_blas_dnrm2(&group_i.vector);
    if(group_i_norm<1)
      gsl_vector_set_zero(&group_i.vector);
    else{
      group_i_norm = 1-1/group_i_norm;
      gsl_vector_scale(&group_i.vector, group_i_norm);
    }
  }
}

void shrink_matrix(gsl_matrix *v, double sigma){
    double vi;
    for(int i=0; i<v->size1; ++i)
        for(int j=0; j<v->size2; ++j){
            vi = gsl_matrix_get(v, i, j);
            if(vi > sigma) { gsl_matrix_set(v, i, j, vi-sigma); }
            else if (vi < -sigma) { gsl_matrix_set(v, i, j, vi+sigma); }
            else { gsl_matrix_set(v, i, j, 0); }
        }
}

void shrink_column_matrix(gsl_matrix *v){
    double column_nrm;
    for(int i=0; i<v->size2; ++i){
        gsl_vector_view temp = gsl_matrix_column(v, i);
        column_nrm = gsl_blas_dnrm2(&temp.vector);
        if(column_nrm<1)
            gsl_vector_set_zero(&temp.vector);
        else{
            column_nrm = 1-1/column_nrm;
            gsl_vector_scale(&temp.vector, column_nrm);
        }
    }
}

void shrink_group_matrix_general(gsl_matrix *v,int *group_split, int*group_split_length){
    double block_nrm;
    for(int i=0; i<((*group_split_length)-1); ++i){
        gsl_matrix_view temp = gsl_matrix_submatrix(v, 0, group_split[i], v->size1, group_split[i+1]-group_split[i]);
        block_nrm = gsl_matrix_Fnorm(&temp.matrix);
        if(block_nrm<1)
            gsl_matrix_set_zero(&temp.matrix);
        else{
            block_nrm = 1-1/block_nrm;
            gsl_matrix_scale(&temp.matrix, block_nrm);
        }
    }
}

void shrink_block_matrix_general(gsl_matrix *v,int *group_split, int*group_split_length){
  double block_nrm;
  for(int i=0; i<((*group_split_length)-1); ++i){
    for(int j=0; j<((*group_split_length)-1); ++j){
      gsl_matrix_view temp = gsl_matrix_submatrix(v, group_split[i],group_split[j], group_split[i+1]-group_split[i],group_split[j+1]-group_split[j]);
      block_nrm = gsl_matrix_Fnorm(&temp.matrix);
      if(block_nrm<1)
        gsl_matrix_set_zero(&temp.matrix);
      else{
        block_nrm = 1-1/block_nrm;
        gsl_matrix_scale(&temp.matrix, block_nrm);
      }
    }
  } 
}

void gsl_matrix_exp(gsl_matrix* X){
    int m = (int)X->size1;
    int n = (int)X->size2;
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j)
            gsl_matrix_set(X, i, j, exp(gsl_matrix_get(X, i, j)));
}

void gsl_vector_log(gsl_vector* v){
    int n = (int)v->size;
    for(int i=0; i<n; ++i)
        gsl_vector_set(v, i, log(gsl_vector_get(v, i)));
}

void gsl_matrix_col_scale(gsl_matrix *X){
  int n = (int) X->size2;
  for(int i=0; i<n; ++i){
    gsl_vector_view temp = gsl_matrix_column(X, i);
    double temp_sum = gsl_blas_dasum(&temp.vector);
    gsl_vector_scale(&temp.vector, 1.0/temp_sum);
  }
}

double gsl_matrix_Fnorm(gsl_matrix* X){
    int m = (int)X->size1;
    int n = (int)X->size2;
    double temp, sum=0.0;
    for(int i=0; i<m; ++i)
        for(int j=0; j<n; ++j){
            temp = gsl_matrix_get(X, i, j);
            if(temp!=0)
                sum += temp * temp;
        }
    return sqrt(sum);
}
