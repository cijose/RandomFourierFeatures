#include <cublas_v2.h>
#include "arena.h"
#include "math_functions.h"
#include <cusolverDn.h>
#include <curand_kernel.h>
#include <cassert>
__global__ void mul_kernel(int N, const float* a,
                           const float* b, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = a[index] * b[index];
  }
}
void gpu_mul(int N, const float* a,
             const float* b, float* y) {
  mul_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

__global__ void div_kernel(int N, const float* a,
                           const float* b, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = a[index] / b[index];
  }
}
void gpu_div(int N, const float* a,
             const float* b, float* y) {
  div_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

__global__ void add_kernel(int N, const float* a,
                           const float* b, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = a[index] + b[index];
  }
}
void gpu_add(int N, const float* a,
             const float* b, float* y) {
  add_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, b, y);
}


__global__ void fill_kernel(int N, const float a,
                            float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = a;
  }
}


__global__ void pow_kernel(int N,  const float* a, const float p,
                            float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = pow(a[index], p);
  }
}
__global__ void clip_kernel(int N,  const float* a, const float p,
                            float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float fabsai = fabs(a[index]);
    y[index] = fabsai < p ? a[index] : p * a[index] / fabsai ;
  }
}

__global__ void addscalar_kernel(int N,  const float a, const float *x,
                            float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    y[index] = x[index] + a;
  }
}


__global__ void apx_kernel(int N,  const float a, float *x) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float xd = x[index * N  + index];
    x[index * N  + index] = xd + float(a);
  }
}


__global__ void apx_complex_kernel(int N,  const float a, float *x) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    int i = 2 * index;
    float xr = x[i * N  + i];
    //float xi = x[i * N  + i + 1];
    x[i * N  + i] = xr + float(a);
    //  x[i * N  + i + 1] = xi + float(1e-5);
  }
}

/*
  Kernel to put a uniform[0, 1] rvs between [a, b]
 */
__global__ void unif_ab_kernel(int N, const float a,
                               const float b, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const float c = b - a;
  if (index < N) {
    y[index] = y[index] * c + a;
  }
}

__global__ void setup_kernel(curandState *state) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(7+id, id, 0, &state[id]);
}

__global__ void chi_distribution_kernel(curandState *state, int N, int DOF, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[index];
  if (index < N) {
    float chi2rv =  0;
    int j = 0;
    for(j = 0; j < DOF; j++) {
      float r = curand_normal(&localState);
      chi2rv += r * r;
    }
    y[index] = sqrt(chi2rv);
  }
}

__global__ void rademacher_distribution_kernel(curandState *state, int N, float* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[index];
  if (index < N) {
    float r = curand_normal(&localState);
    y[index] = r > 0 ? float(1) : float(-1);
  }
}

void gpu_chi(int N,  int dof, float *x) {
  curandState *devStates;
  CUDA_CHECK(cudaMalloc((void **)&devStates, CUDA_GET_BLOCKS(N) * CUDA_NUM_THREADS * sizeof(curandState)));
  setup_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(devStates);
  chi_distribution_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(devStates, N, dof, x);
  CUDA_CHECK(cudaFree(devStates));
}

void gpu_rademacher(int N, float *x) {
  curandState *devStates;
  CUDA_CHECK(cudaMalloc((void **)&devStates, CUDA_GET_BLOCKS(N) * CUDA_NUM_THREADS * sizeof(curandState)));
  setup_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(devStates);
  rademacher_distribution_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(devStates, N,  x);
  CUDA_CHECK(cudaFree(devStates));
}

__global__ void gpu_permute_kernel(int M , int  N, const float *PI,
                                   const float* X, float* Y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < M * N) {
    const int n = index % N;
    const int m = index / N;
    Y[index] = X[N * m + int(PI[n])];
  }
}


__global__ void log_kernel(int N, const float* in,
                           float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    out[index] = log(in[index]);
  }
}

__global__ void sqrt_kernel(int N, const float* in,
                           float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    out[index] =  sqrt(in[index]);
  }
}

__global__ void exp_kernel(int N, const float* in,
                           float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    out[index] = exp(in[index]);
  }
}
__global__ void tanh_forward_kernel(int N, const float* in,
                            float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float exp2x = exp(2 * in[index]);
    out[index] = (exp2x - 1) / (exp2x + 1);
  }
}
__global__ void sin_kernel(int N, const float* in,
                            float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    out[index] = sin(in[index]);
  }
}
__global__ void cos_kernel(int N, const float* in,
                            float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    out[index] = cos(in[index]);
  }
}
__global__ void tanh_backward_kernel(int N, const float* in_data, const float* in_diff,
                                     float* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float exp2x = in_data[index];
    out_diff[index] = in_diff[index] * (1 - exp2x * exp2x);
  }
}

__global__ void sigmoid_forward_kernel(int N, const float* in,
                            float* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float expx = exp(-in[index]);
    out[index] =  float(1) / (expx + float(1));
  }
}
__global__ void sigmoid_backward_kernel(int N, const float* in_data, const float* in_diff,
                                     float* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    float expx = in_data[index];
    out_diff[index] = in_diff[index] * (expx - expx * expx);
  }
}


__global__ void diag_gemm_kernel(int MN, int N, float alpha, const float* X,
                                 const float* W, float beta, float *Y ) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < MN) {
    const int n = index % N;
    Y[index] = alpha * X[index] * W[n] + beta * Y[index];
  }  
}




void gpu_diag_gemm(int M, int N, float alpha, const float* X,
                   const float* W, float beta, float *Y ) {

  int MN = M * N;
  diag_gemm_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(MN, N, alpha, X, W, beta, Y);
}

__global__ void cm_intrleav_forward_kernel(int M, int N, const float* X,
                                           const float* b, float* Y) {
  int MN = M * N;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < MN) {
    int i =  2 * index;
    float realXn = X[i];
    float imagXn = X[i + 1];
    float epsilon  = 1e-5;
    float absz  = float(1) / (epsilon + sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)));
    float bn = b[index % N];
    Y[i] = 0;
    Y[i + 1] = 0;
    float tabsz = float(1) + bn * absz;
    if(tabsz  > 0)
    {
      Y[i] = realXn * tabsz;
      Y[i + 1] = imagXn * tabsz;
    }
  }
}

__global__ void cm_intrleav_backward_kernel(int M, int N, const float* X,
                                            const float* b, const float* gradY,
                                            float* gradX, float* gradb) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int MN = M * N;
  if(index < MN) {
    float epsilon = 1e-5;
    int i =  2 * index;
    float realXn = X[i];
    float imagXn = X[i + 1];
    float bn = b[index % N];
    float absz  = 1 / (sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)) + epsilon);
    gradX[i] = 0;
    gradX[i + 1] = 0;
    float tabsz = float(1) + bn * absz;
    if(tabsz > 0)
    {
       const float realgradYn = gradY[i];
       const float imaggradYn = gradY[i + 1];
       float gradbn = (realXn * realgradYn + imaggradYn * imagXn) * absz;
       atomicAdd(&gradb[index % N], gradbn);
       float k1 = bn * absz;
       float abszcub = pow(absz, 3);
       float k1k2 = bn * realXn * abszcub;
       float realdXn = 1 + k1 - k1k2 * realXn;
       float imagdXn = -imagXn * k1k2;
       gradX[i] = realdXn * realgradYn + imagdXn * imaggradYn;
       k1k2 =  bn * imagXn * abszcub ;
       realdXn = -realXn * k1k2;
       imagdXn = 1 + k1 - k1k2 * imagXn;
       gradX[i + 1] =  realdXn * realgradYn + imagdXn * imaggradYn;
    }
  }
}

void gpu_complex_modulus_interleaved_relu(int M, int N, const float *X,
                                          const float* b, float* Y) {
 int MN = M * N;
 cm_intrleav_forward_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(M, N, X,
                                                                       b, Y);
}

void gpu_complex_modulus_interleaved_drelu(int M, int N, const float  *X,
                                           const float* b, const float* gradY,
                                           float* gradX, float* gradb) {
  int MN = M * N;
  cm_intrleav_backward_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(M, N, X,
                                                                         b,
                                                                         gradY,
                                                                         gradX,
                                                                         gradb);
}


__global__ void cm_intrleav_approxtanh_forward_kernel(int M, int N, const float* X,
                                                      const float* b, float* Y) {
  int MN = M * N;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < MN) {
    int n =  2 * index;
    float realXn = X[n];
    float imagXn = X[n + 1];
    float realb = b[n];
    float imagb = b[n + 1];
    realXn += realb;
    imagXn += imagb;
    float absz  = sqrt(pow(realXn, 2) + pow(imagXn, 2));
    absz = absz < 50 ? absz : 50;
    float tabsz = 2 * (1 - exp(-absz)) - 1;
    Y[n] = realXn * tabsz;
    Y[n + 1] = imagXn * tabsz;
  }
}

__global__ void cm_intrleav_approxtanh_backward_kernel(int M, int N, const float* X,
                                                       const float* b, const float* gradY,
                                                       float* gradX, float* gradb) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int MN = M * N;
  if(index < MN) {
    int n =  2 * index;
    float realb = b[n];
    float imagb = b[ n + 1];
    float realbXn = X[n] + realb;
    float imagbXn = X[n + 1] + imagb;
    float realbXn2 = pow(realbXn, 2);
    float imagbXn2 = pow(imagbXn, 2);
    float absz  = sqrt(pow(realbXn, 2) + pow(imagbXn, 2));
    absz = absz < 50 ? absz : 50;
    float expabsz = exp(-absz);
    float tabsz = 2 * (1 - expabsz) - 1;
    const float realgradYn = gradY[n];
    const float imaggradYn = gradY[n + 1];
    float K1 = 2 *  expabsz / absz;
    float realdbn = tabsz + realbXn2 * K1;
    float imagdbn = realbXn * imagbXn * K1;
    float realgradbn = (realdbn * realgradYn + imagdbn * imaggradYn);
    realdbn = imagdbn;
    imagdbn = tabsz +  imagbXn2 * K1;
    float imaggradbn = (realdbn * realgradYn + imagdbn * imaggradYn);
    atomicAdd(&gradb[n], realgradbn);
    atomicAdd(&gradb[n + 1], imaggradbn);
    gradX[ n] = realgradbn;
    gradX[n + 1] = imaggradbn;    
  }
}


void gpu_complex_modulus_interleaved_approxtanh(int M, int N, const float *X,
                                                const float* b, float* Y) {
 int MN = M * N;
 cm_intrleav_approxtanh_forward_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(M, N, X,
                                                                                 b, Y);
}

void gpu_complex_modulus_interleaved_dapproxtanh(int M, int N, const float  *X,
                                                 const float* b, const float* gradY,
                                                 float* gradX, float* gradb) {

  int MN = M * N;
  cm_intrleav_approxtanh_backward_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(M, N, X,
                                                                                    b,
                                                                                    gradY,
                                                                                    gradX,
                                                                                    gradb);
}


__global__ void gpu_hadamard_kernel1(int MN, float* Y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < MN - 1 && (index % 2 == 0)) {
    float T1 = Y[index];
    float T2 = Y[index + 1];
    Y[index] = T1 + T2;
    Y[index + 1] = T1 - T2; 
  }
}

__global__ void gpu_hadamard_kernel2(int MN, int bit, float* Y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < MN && !(bit & index)) {
    int j = index | bit;
    float T1 = Y[index];
    float T2 = Y[j];
    Y[index] = T1  + T2;
    Y[j] = T1  - T2;
  }
}

void gpu_rearrange(int batch_size, int input_dim, int num_blocks,
                  const float *T, float *X) {
  for(int  b =0; b < batch_size; b++) {
    float *X_b = X + b * num_blocks * input_dim;
    for(int n = 0; n < num_blocks; n++) {
      CUDA_CHECK(cudaMemcpy(X_b + n * input_dim, T + n * batch_size * input_dim + b * input_dim, sizeof(float) * input_dim, cudaMemcpyDeviceToDevice));
    }
  }
} 

void gpu_hadamard(int M, int N, float* Y) {
  bool ispow2 = !(N == 0) && !(N & (N - 1));
  assert(ispow2 == true);
  gpu_hadamard_kernel1<<<CUDA_GET_BLOCKS(M * N), CUDA_NUM_THREADS>>>(M * N, Y);
  for(int bit = 2; bit < N; bit <<= 1) {
    gpu_hadamard_kernel2<<<CUDA_GET_BLOCKS(M * N), CUDA_NUM_THREADS>>>(M * N, bit, Y);
  }
}

void gpu_real_svd(int M, int N, float *A,
                  float *U, float *S, float *V) {
  int work_size;
  int *devInfo, devInfo_h;
  CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);
  cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
  float *work;
  CUDA_CHECK(cudaMalloc(&work, work_size * sizeof(float)));
  cusolverDnSgesvd(solver_handle, 'A', 'A', M, N, A, M, S, U, M, V, N, work, work_size, NULL, devInfo);
  CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h < 0)
    throw cervnet_error("SVD unsuccessfull for the full SVD calculation. Parameters are wrong\n");
  else if(devInfo_h > 0)
    throw cervnet_error("SVD unsuccessfull for the full SVD calculation. The number of  superdiagonals of an intermediate bidiagonal form did not converge to zero\n");
  cusolverDnDestroy(solver_handle);
  cudaFree(work);
  cudaFree(devInfo);
}

void gpu_complex_svd(int M, int N, float *A,
                     float *U, float *S, float *V) {
  int work_size;
  int *devInfo, devInfo_h;
  CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);
  cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
  float2 *work;
  CUDA_CHECK(cudaMalloc(&work, work_size * sizeof(float2)));
  cusolverDnCgesvd(solver_handle, 'A', 'A', M, N, (float2*)A, M, S, (float2*)U, M, (float2*)V, N, work, work_size, NULL, devInfo);
  CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
  if (devInfo_h < 0)
    throw cervnet_error("SVD unsuccessfull for the full SVD calculation. Parameters are wrong\n");
  else if(devInfo_h > 0)
    throw cervnet_error("SVD unsuccessfull for the full SVD calculation. The number of  superdiagonals of an intermediate bidiagonal form did not converge to zero\n");
  cusolverDnDestroy(solver_handle);
  cudaFree(work);
  cudaFree(devInfo);
}

void gpu_random_orthogonal_matrix(int N, float *W) {
  float *U_gpu = NULL;
  float *S_gpu = NULL;
  float *V_gpu = NULL;
  gpu_normal(N * N, 0, 1.0, W);
  CUDA_CHECK(cudaMalloc(&U_gpu, N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&S_gpu,  N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V_gpu,  N * N * sizeof(float)));
  gpu_real_svd(N, N, W, U_gpu, S_gpu, V_gpu);
  gpu_gemm(CblasNoTrans, CblasTrans, N, N, N,
           1, U_gpu, V_gpu, 0, W);
  cudaFree(U_gpu);
  cudaFree(S_gpu);
  cudaFree(V_gpu);
}

void gpu_random_unitary_matrix(int N, float *W) {
  float *U_gpu = NULL;
  float *S_gpu = NULL;
  float *V_gpu = NULL;
  gpu_normal(2 * N * N, 0, 1.0, W);
  CUDA_CHECK(cudaMalloc(&U_gpu, 2 * N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V_gpu, 2 * N * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&S_gpu,  N * sizeof(float)));
  gpu_complex_svd(N, N, W, U_gpu, S_gpu, V_gpu);
  gpu_complex_gemm(CblasNoTrans, CblasTrans, N, N, N,
                   1, U_gpu, V_gpu, 0, W);
  cudaFree(U_gpu);
  cudaFree(S_gpu);
  cudaFree(V_gpu);

}

void gpu_initialize_unitary(int NF, int* factor_sizes,  float *W) {
  float *Wf = W;
  for(int f = 0; f < NF; f++) {
    int N = factor_sizes[f];
    gpu_random_unitary_matrix(N, Wf);
    Wf = Wf + int(2 * pow(factor_sizes[f], 2));
  }
}

void gpu_initialize_orthogonal(int NF, int* factor_sizes,  float *W) {
  float *Wf = W;
  for(int f = 0; f < NF; f++) {
    int N = factor_sizes[f];
    gpu_random_orthogonal_matrix(N, Wf);
    Wf = Wf + int(pow(factor_sizes[f], 2));
  }
}

void gpu_fill(int N, const float a, float* y) {
  fill_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, a, y);
}

void gpu_lecun98(int N, int fan_in, float *x) {
  CURAND_CHECK(curandGenerateUniform(Arena::curand_generator(), x, N));
  float range = sqrt(float(3.)) / float(2 * fan_in);
  unif_ab_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, -range, range,  x);
}

void gpu_glorot10(int fan_in, int fan_out,  float *x) {
  int N = fan_in * fan_out;
  float range = sqrt(float(6.) / float(fan_in + fan_out));
  CURAND_CHECK(curandGenerateUniform(Arena::curand_generator(), x, N));
  unif_ab_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, -range, range,  x);
}

void gpu_uniform(int N, float r1, float r2, float *x) {
  CURAND_CHECK(curandGenerateUniform(Arena::curand_generator(), x, N));
  unif_ab_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, r1, r2,  x);
}



void gpu_permute(int M , int  N, const float *PI,
                 const float* X, float* Y) {
  int MN = M * N;
  gpu_permute_kernel<<<CUDA_GET_BLOCKS(MN), CUDA_NUM_THREADS>>>(M, N, PI, X, Y);
}

void gpu_he15(int N, int fan_in, float *x) {
  float stddev = sqrt(float(2.) / float(fan_in));
  CURAND_CHECK(curandGenerateNormal(Arena::curand_generator(), x, N, float(0), stddev));
}

void gpu_normal(const  int N, float mean, float stddev, float *x) {
  CURAND_CHECK(curandGenerateNormal(Arena::curand_generator(), x, N, mean, stddev));
}


void gpu_permutation_array(int N, float* y) {
  float *x = NULL;
  CUDA_CHECK(cudaMallocHost(&x, N * sizeof(float)));
  for(int i = 0; i < N; i++) {
    x[i] = float(i);
  }
  for(int i = 0; i < N; i++) {
    int r =  Arena::rng_stream().randi(i, N);
    int tmp = x[r];
    x[r] = x[i];
    x[i] = tmp;
  }
  CUDA_CHECK(cudaMemcpy(y, x, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaFreeHost(x)); 
}
void gpu_allzero(int N, float *x) {
  fill_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(
      N, float(0), x);
}

void gpu_log(int N, const float* X, float* Y) {
  log_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}


void gpu_pow(int N, const float* X, const float p,  float* Y) {
  pow_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, p, Y);
}

void gpu_clip(int N, const float* X, const float p,  float* Y) {
  clip_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, p, Y);
}

void gpu_addscalar(int N, const float a, const float* X,  float* Y) {
  addscalar_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, X, Y);
}

void gpu_sqr(int N, const float* X, float* Y) {
  sqrt_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}

void gpu_exp(int N, const float* X, float* Y) {
  exp_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}

void gpu_tanh(int N, const float* X, float* Y) {
  tanh_forward_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}


void gpu_sin(int N, const float* X, float* Y) {
  sin_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}

void gpu_cos(int N, const float* X, float* Y) {
  cos_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}


void gpu_dtanh(int N, const float* X, const float* Z,  float* Y) {
  tanh_backward_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Z, Y);
}

void gpu_sigmoid(int N, const float* X, float* Y) {
  sigmoid_forward_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Y);
}

void gpu_dsigmoid(int N, const float* X, const float* Z,  float* Y) {
  sigmoid_backward_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, X, Z, Y);
}

void gpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, int M, int N, int K,
              const float alpha, const float* A, const float* B, const float beta,
              float* C) {
  // Note that cublas follows fortran (column major like matlab) order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Arena::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));

}

void gpu_complex_gemm(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                      const float alpha, const float* A, const float* B, const float beta,
                      float* C) {
  //Remeber complex arithmetic """conjugate transpose"""
  // Note that cublas follows fortran (column major like matlab) order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_C;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_C;
  float a[2] = {alpha, 0};
  float b[2] = {beta, 0};
  CUBLAS_CHECK(cublasCgemm(Arena::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, (float2*)a, (float2*)B, ldb, (float2*)A,
                           lda, (float2*)b, (float2*)C, N));

}

//The matrix A is real
void gpu_real_complex_gemm(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                           const float alpha, const float* A, const float* B, const float beta,
                           float* C) {
  int D =2 * M * K;
  float *T = NULL;
  CUDA_CHECK(cudaMalloc(&T, D * sizeof(float)));
  CUDA_CHECK(cudaMemset(T, 0, D * sizeof(float)));
  const float unity = 1;
  CUBLAS_CHECK(cublasSaxpy(Arena::cublas_handle(), D / 2, &unity, A, 1, T, 2));
  // Note that cublas follows fortran (column major like matlab) order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_C;
  float a[2] = {alpha, 0};
  float b[2] = {beta, 0};
  CUBLAS_CHECK(cublasCgemm(Arena::cublas_handle(), cuTransB, cuTransA,
                           N, M, K, (float2*)a, (float2*)B, ldb, (float2*)T,
                           lda, (float2*)b, (float2*)C, N));
  CUDA_CHECK(cudaFree(T));
}


void gpu_geam(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
             int M, int N, const float *alpha, const float *A,
             const float *beta, const float *B, float *C) {

/*
  To do
*/
}

void gpu_gemv(const CBLAS_TRANSPOSE TransA, int M,
              int N, const float alpha, const float* A, const float* x,
              const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Arena::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

void gpu_axpy(int N, const float alpha, const float* X,
              float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Arena::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void gpu_complex_axpy(int N, const float alpha, const float* X,
                      float* Y) {
  float a[2] = {alpha, 0};
  CUBLAS_CHECK(cublasCaxpy(Arena::cublas_handle(), N, (float2*)a, (float2*)X,
                           1, (float2*)Y, 1));
}

void gpu_copy(int N, const float* X, float* Y) {
  CUBLAS_CHECK(cublasScopy(Arena::cublas_handle(), N, X, 1, Y, 1));
}

void gpu_scal(int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Arena::cublas_handle(), N, &alpha, X, 1));
}

void gpu_axpby(int N, const float alpha, const float* X,
    const float beta, float* Y) {
  gpu_scal(N, beta, Y);
  gpu_axpy(N, alpha, X, Y);
}

float gpu_dot(int n, const float* x, const float* y) {
  float out;
  CUBLAS_CHECK(cublasSdot(Arena::cublas_handle(), n, x, 1, y, 1, &out));
  return out;
}

float gpu_nrm2(int n, const float* x) {
  float out;
  CUBLAS_CHECK(cublasSnrm2(Arena::cublas_handle(), n, x, 1, &out));
  return out;
}

void gpu_apx(int N, float a,  const float* X, float* Y) {
  gpu_copy(N * N, X, Y);
  apx_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, Y);
}

void gpu_complex_apx(int N, float a,  const float* X, float* Y) {
  gpu_copy(2 * N * N, X, Y);
  apx_complex_kernel<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(N, a, Y);
}
