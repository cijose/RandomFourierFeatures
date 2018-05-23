#ifndef _MATH_FUNCTIONS_H_
#define _MATH_FUNCTIONS_H_

extern "C" {
#include <cblas.h>
}
#include <math.h>
#include <xmmintrin.h>
//#include <omp.h>
#include <cublas_v2.h>
#include "arena.h"

void cpu_rearrange(int batch_size, int input_dim, int num_blocks,
                  const float *T, float *X);
void gpu_rearrange(int batcch_size, int input_dim, int num_blocks,
                  const float *T, float *X);
    
void svd2x2(const float a[4], float u[4], float s[2], float v[4]);
void matmulcomplex2x2(const float A[8], const float B[8], float C[8]);
void conjugate_transpose2x2(float M[8]);
void conjugate_transpose2x2(const float M[8], float R[8]);
void cpu_complex_modulus_interleaved_relu(int M, int N, const float *X,
                                          const float* b, float* Y);

void cpu_permute_forward(int N,  const float*X, const int* PI,
                         float* Y);

void cpu_permute_backward(int N,  const float*X, const int* PI,
                          float* Y);


void cpu_diag_gemm(int M, int N, float alpha,  const float* X,
                   const float* W, float beta, float *Y );
void gpu_diag_gemm(int M, int N, float alpha, const float* X,
                   const float* W, float beta, float *Y );

void cpu_permute(int M, int N, const float* PI,
                const float* X, float* Y);

void gpu_permute(int M, int N, const float* PI,
                const float* X, float* Y);

float cpu_relative_error(int N,  const float* grad_ptr,
                         const float* numer_grad_ptr);


void gpu_real_svd(int M, int N, float *A,
                 float *U, float *S, float *V);
void gpu_unitary_svd(int M, int N, float *A,
                 float *U, float *S, float *V);
void gpu_random_orthogonal_matrix(int N, float *W);
void gpu_random_unitary_matrix(int N, float *W);

void cpu_initialize_svd2x2(int N, float* W);
void cpu_initialize_complexsvd2x2(int N, float* W);
void cpu_initialize_unitary(int NF, int* factor_sizes,  float *W);
void cpu_initialize_orthogonal(int NF, int* factor_sizes,  float *W);
void gpu_initialize_unitary(int NF, int* factor_sizes,  float *W);
void gpu_initialize_orthogonal(int NF, int* factor_sizes,  float *W);
void cpu_initialize_glorot10(int NF, int* factor_sizes,  float *W);
float cpu_complexdet2x2(const float W[8], float D[2]);
void cpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, int M, int N, int K,
              const float alpha, const float* A, const float* B, const float beta,
              float* C);
void gpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, int M, int N, int K,
              const float alpha, const float* A, const float* B, const float beta,
              float* C);

void cpu_complex_gemm(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                      const float alpha, const float* A, const float* B,
                      const float beta, float* C);

void cpu_real_complex_gemm(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C);


void gpu_complex_gemm(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                      const float alpha, const float* A, const float* B,
                      const float beta, float* C);

void gpu_real_complex_gemm(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C);

//Y = X + a * I
void gpu_apx(int N, float a,  const float* X, float* Y);

void gpu_complex_apx(int N, float a,  const float* X, float* Y);

void gpu_geam(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
              int m, int n, const float *alpha, const float *A,
              const float *beta, const float *B, float *C);

void cpu_gemv(const CBLAS_TRANSPOSE TransA, int M, int N,
              const float alpha, const float* A, const float* x, const float beta,
              float* y);
void gpu_gemv(const CBLAS_TRANSPOSE TransA, int M, int N,
              const float alpha, const float* A, const float* x, const float beta,
              float* y);
void cpu_axpy(int N, const float alpha, const float* X,
              float* Y);
void gpu_axpy(int N, const float alpha, const float* X,
              float* Y);
void gpu_complex_axpy(int N, const float alpha, const float* X,
                      float* Y);
void cpu_addscalar(int N, const float alpha, const float* X,
                   float* Y);
void gpu_addscalar(int N, const float alpha, const float* X,
                   float* Y);
void cpu_axpby(int N, const float alpha, const float* X,
               const float beta, float* Y);
void gpu_axpby(int N, const float alpha, const float* X,
               const float beta, float* Y);
void cpu_copy(int N, const float *X, float *Y);
void gpu_copy(int N, const float *X, float *Y);
void cpu_scal(int N, const float alpha, float *X);
float cpu_max(int N, const float *X);
int cpu_max_index(int N, const float *X);
void cpu_exp(int N, const float *X, float *Y);
void gpu_exp(int N, const float *X, float *Y);
void cpu_dexp(int N, const float *in_data, const float *in_diff, float *out);
void cpu_tanh(int N, const float *X, float *Y);
void gpu_tanh(int N, const float *X, float *Y);

void cpu_sin(int N, const float *X, float *Y);
void gpu_sin(int N, const float *X, float *Y);

void cpu_cos(int N, const float *X, float *Y);
void gpu_cos(int N, const float *X, float *Y);

void cpu_dtanh(int N, const float *in_data, const float *in_diff, float *out);
void gpu_dtanh(int N, const float *in_data, const float *in_diff, float *out);

void cpu_sigmoid(int N, const float *X, float *Y);
void gpu_sigmoid(int N, const float *X, float *Y);
void cpu_dsigmoid(int N, const float *in_data, const float *in_diff, float *out);
void gpu_dsigmoid(int N, const float *in_data, const float *in_diff, float *out);

void cpu_hadamard(int M, int N, float *X);
void gpu_hadamard(int M, int N, float *X);



void cpu_complex_modulus_relu(int M, int N, const float  *X, const float* b, float* Y);
void cpu_complex_modulus_drelu(int M, int N, const float  *X,
                               const float* b, const float* gradY, float* gradX,
                               float* gradb);
void cpu_complex_modulus_tanh(int M, int N, const float  *X, const float* b, float* Y);
void cpu_complex_modulus_dtanh(int M, int N, const float  *X,
                               const float* b, const float* gradY, float* gradX,
                               float* gradb);
/*
    From Efficient backprop, Yann Lecun, Leon Bottou 1998
*/
void cpu_lecun98(int N, int fan_in, float *x);
void gpu_lecun98(int N, int fan_in, float *x);
/*
    From Understanding the diffcuilt of traing neural nets, Xavier Glorot, Yoshua Bengio : AISTATS 2010
*/
void cpu_glorot10(int fan_in, int fan_out, float *x);
void gpu_glorot10(int fan_in, int fan_out, float *x);
/*
    From Delving deep into rectifiers, Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun:  Arxiv 2015
*/

void cpu_he15(int N, int fan_in, float *x);
void cpu_allzero(int N, float *x);
void cpu_normal(int N, const float mean, const float stddev, float *x);
void cpu_uniform(int N, const float r1, const float r2, float *x);
void gpu_uniform(int N, const float r1, const float r2, float *x);
void gpu_he15(int N, int fan_in, float *x);
void gpu_allzero(int N, float *x);
void gpu_normal(int N, const float mean, const float stddev, float *x);
void cpu_log(int N, const float *X, float *Y);
void gpu_log(int N, const float *X, float *Y);
void cpu_sqr(int N, const float *X, float *Y);
void gpu_sqr(int N, const float *X, float *Y);
void cpu_abs(int N, const float *X, float *Y);
void cpu_clip(int N, const float *X, const float p, float *Y);
void gpu_clip(int N, const float *X, const float p, float *Y);
void cpu_pow(int N, const float *X, const float p, float *Y);
void gpu_pow(int N, const float *X, const float p, float *Y);
void gpu_scal(int N, const float alpha, float *X);
float cpu_dot(int n, const float* x, const float* y);
void cpu_add(int N, const float* a, const float* b, float* y);
void gpu_add(int N, const float* a, const float* b, float* y);
void gpu_mul(int N, const float* a, const float* b, float* y);
void cpu_mul(int N, const float* a, const float* b, float* y);
void gpu_div(int N, const float* a, const float* b, float* y);
void cpu_div(int N, const float* a, const float* b, float* y);
void gpu_fill(int N, const float a, float* y);
void cpu_fill(int N, const float a, float* y);
float cpu_max(int N, const float* y);
float cpu_nrm2(int N, const float* y);
float gpu_nrm2(int N, const float* y);
float cpu_min(int N, const float* y);
float gpu_dot(int n, const float* x, const float* y);


void cpu_chi(int N, int dof, float* x);
void gpu_chi(int N, int dof, float* x);

void cpu_rademacher(int N, float* x);
void gpu_rademacher(int N, float* x);

void cpu_permutation_array(int N, float* x);
void gpu_permutation_array(int N, float* x);


inline void cblas_saxpby(int N, const float alpha, const float* X,
                         int incX, const float beta, float* Y,
                         int incY) {
  cblas_sscal(N, beta, Y, incY);
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}


#endif
