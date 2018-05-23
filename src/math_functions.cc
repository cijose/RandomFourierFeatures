
#include "math_functions.h"
#include <algorithm>
#include <cassert>

void cpu_rearrange(int batch_size, int input_dim, int num_blocks,
                   const float *T, float *X) {
  for(int  b =0; b < batch_size; b++) {
    float *X_b = X + b * num_blocks * input_dim;
    for(int n = 0; n < num_blocks; n++) {
      memcpy(X_b + n * input_dim, T + n * batch_size * input_dim + b * input_dim, sizeof(float) * input_dim);
    }
  }
}

void cpu_diag_gemm(int M, int N, float alpha, const float* X,
                   const float* W, float beta, float *Y ) {
  for(int m =0; m < M; m++){
    const float *X_m  = X + m * N;
    float *Y_m  = Y + m * N;
    for(int n = 0; n < N; n++) {
      Y_m[n] = alpha * X_m[n] * W[n] + beta * Y_m[n];
    }
  }
}

void cpu_permute_forward(int N, const float*X, const int* PI,
                         float* Y){
  for (size_t n = 0; n <  N; n++) {
    Y[n] = X[PI[n]];
  }
}

void cpu_permute_backward(int N,  const float*X, const int* PI,
                          float* Y){
  for (size_t n = 0; n <  N; n++) {
    Y[PI[n]] = X[n];
  }
}




/*
Need to optimize
*/
//A = Udiag(S)V.T
void svd2x2(const float a[4], float u[4], float s[2], float v[4]) {
  s[0] = (sqrt(pow(a[0] - a[3], 2) + pow(a[1] + a[2], 2)) +
          sqrt(pow(a[0] + a[3], 2) + pow(a[1] - a[2], 2))) / 2;
  s[1] = fabs(s[0] - sqrt(pow(a[0] - a[3], 2) + pow(a[1] + a[2], 2)));
  if(s[0] > s[1]) {
    v[2] =  sin((atan2(2 * (a[0] * a[1] + a[2] * a[3]), a[0] * a[0]
                       - a[1] * a[1] + a[2] * a[2] - a[3] * a[3])) / 2);
  }
  else{
    v[2] =  0;
  }
  v[0] = sqrt(1 - v[2] * v[2]);
  v[1] = -v[2];
  v[3] = v[0];
  u[0] = (s[0] != 0) ? (a[0] * v[0] + a[1] * v[2]) / s[0] : 1;
  u[2] = (s[0] != 0) ? (a[2] * v[0] + a[3] * v[2]) / s[0] : 0;
  u[1] = (s[1] != 0) ? (a[0] * v[1] + a[1] * v[3]) / s[1] : -u[2];
  u[3] = (s[1] != 0) ? (a[2] * v[1] + a[3] * v[3]) / s[1] : u[0];
  float a1 = a[0], b = a[1], c = a[2], d = a[3];
  float p = u[0] * s[0] * v[0] + u[1] * s[1] * v[1];
  float q = u[0] * s[0] * v[2] + u[1] * s[1] * v[3];
  float r = u[2] * s[0] * v[0] + u[3] * s[1] * v[1];
  float s1 = u[2] * s[0] * v[2] + u[3] * s[1] * v[3];
}

void matmulcomplex2x2(const float A[8], const float B[8], float C[8]) {
  float rA11 = A[0], iA11 = A[1];
  float rA12 = A[2], iA12 = A[3];
  float rA21 = A[4], iA21 = A[5];
  float rA22 = A[6], iA22 = A[7];

  float rB11 = B[0], iB11 = B[1];
  float rB12 = B[2], iB12 = B[3];
  float rB21 = B[4], iB21 = B[5];
  float rB22 = B[6], iB22 = B[7];

  C[0] = rA11 * rB11 - iA11 * iB11 + rA12 * rB21 - iA12 * iB21;
  C[1] = rA11 * iB11 + iA11 * rB11 + rA12 * iB21 + iA12 * rB21;

  C[2] = rA11 * rB12 - iA11 * iB12 + rA12 * rB22 - iA12 * iB22;
  C[3] = rA11 * iB12 + iA11 * rB12 + rA12 * iB22 + iA12 * rB22;

  C[4] = rA21 * rB11 - iA21 * iB11 + rA22 * rB21 - iA22 * iB21;
  C[5] = rA21 * iB11 + iA21 * rB11 + rA22 * iB21 + iA22 * rB21;

  C[6] = rA21 * rB12 - iA21 * iB12 + rA22 * rB22 - iA22 * iB22;
  C[7] = rA21 * iB12 + iA21 * rB12 + rA22 * iB22 + iA22 * rB22;

}

void rotation2x2(float theta, float R[8]) {
  memset(R, 0, 8 * sizeof(float));
  float costheta = cos(theta);
  float sintheta = sin(theta);
  R[0] = costheta, R[2] = -sintheta;
  R[4] = sintheta, R[6] = costheta;
}

/*
  P = M * diag([conjugae(A) / modulus(A), conjugate(B) / modulus(B)])
  T = diag([conjugae(A) / modulus(A), conjugate(B) / modulus(B)])
*/
void cancel_phase(bool iscolumn, const float A[2], const float B[2],
                  const float M[8], float P[8], float T[8]) {
  float AC[2], BC[2];
  float modA = sqrt(pow(A[0], 2) + pow(A[1], 2));
  float modB = sqrt(pow(B[0], 2) + pow(B[1], 2));
  AC[0] = A[0] / modA, AC[1] = -A[1] / modA;
  BC[0] = B[0] / modB, BC[1] = -B[1] / modB;
  memset(T, 0, 8 * sizeof(float));
  T[0] = AC[0], T[1] = AC[1];
  T[6] = BC[0], T[7] = BC[1];
  float rM11 = M[0], iM11 = M[1];
  float rM12 = M[2], iM12 = M[3];
  float rM21 = M[4], iM21 = M[5];
  float rM22 = M[6], iM22 = M[7];
  P[0] = rM11 * AC[0] - iM11 * AC[1];
  P[1] = rM11 * AC[1] + iM11 * AC[0];
  if(iscolumn) {
    P[4] = rM21 * AC[0] - iM21 * AC[1];
    P[5] = rM21 * AC[1] + iM21 * AC[0];

    P[2] = rM12 * BC[0] - iM12 * BC[1];
    P[3] = rM12 * BC[1] + iM12 * BC[0];
  }
  else {
    P[2] = rM12 * AC[0] - iM12 * AC[1];
    P[3] = rM12 * AC[1] + iM12 * AC[0];
    P[4] = rM21 * BC[0] - iM21 * BC[1];
    P[5] = rM21 * BC[1] + iM21 * BC[0];
  }
  P[6] = rM22 * BC[0] - iM22 * BC[1];
  P[7] = rM22 * BC[1] + iM22 * BC[0];
}

void conjugate_transpose2x2(float M[8]) {
  float rM11 = M[0], iM11 = M[1];
  float rM12 = M[2], iM12 = M[3];
  float rM21 = M[4], iM21 = M[5];
  float rM22 = M[6], iM22 = M[7];
  M[1] = -iM11;
  M[7] = -iM22;
  M[2] = rM21;
  M[3] = -iM21;
  M[4] = rM12;
  M[5] = -iM12;
}


void conjugate_transpose2x2(const float M[8], float R[8]) {
  memcpy(R, M, 8 * sizeof(float));
  conjugate_transpose2x2(R);
}


void print2x2complex(const float A[8]) {
  std::cout<<A[0]<<"+i"<<A[1]<<" "<<A[2]<<"+i"<<A[3]<<std::endl<<A[4]<<"+i"<<A[5]<<" "<<A[6]<<"+i"<<A[7];
  std::cout<<std::endl<<std::endl;

}

void complexsvd2x2(const float A[8], float U[8], float S[2], float V[8]) {

  float* M = new float [8];
  float* P = new float [8];
  float* Q = new float [8];
  float* R = new float [8];
  float* T = new float [8];

  cancel_phase(true, A, A + 2, A, M, P);

  rotation2x2(atan2(M[2], M[0]), R);
  matmulcomplex2x2(M, R, M);
  cancel_phase(true, M + 4, M + 6, M, M, Q);
  float L[2] = {1, 0};
  cancel_phase(false, M, L, M, M, T);
  float *realM = new float [4];
  float *realU = new float [4];
  float *realV = new float [4];

  realM[0] = M[0], realM[1] = M[2];
  realM[2] = M[4], realM[3]= M[6];
  svd2x2(realM, realU, S, realV);
  memset(U, 0, sizeof(float) * 8);
  memset(V, 0, sizeof(float) * 8);

  U[0] = realU[0], U[2] = realU[1];
  U[4] = realU[2], U[6] = realU[3];

  V[0] = realV[0], V[2] = realV[1];
  V[4] = realV[2], V[6] = realV[3];

  conjugate_transpose2x2(P);
  conjugate_transpose2x2(Q);
  conjugate_transpose2x2(R);
  conjugate_transpose2x2(T);
  conjugate_transpose2x2(V);

  matmulcomplex2x2(T, U, U);
  matmulcomplex2x2(V, Q, V);
  matmulcomplex2x2(V, R, V);
  matmulcomplex2x2(V, P, V);

  memset(M, 0, sizeof(float) * 8);
  M[0] = S[0];
  M[6] = S[1];
  matmulcomplex2x2(U, M, R);
  matmulcomplex2x2(R, V, R);
  delete [] M;
  delete [] P;
  delete [] Q;
  delete [] R;
  delete [] T;
  delete [] realM;
  delete [] realU;
  delete [] realV;
}

void cpu_initialize_svd2x2(int N, float* W) {
  int M = log2(N);
  float* U = new float[4];
  float* S = new float[2];
  float* V = new float[4];
  float stddev = sqrt(float(3.) / float(N));
  for(int i = 0; i < 4 * M; i++) {
    W[i] = float(Arena::rng_stream().uniform_rng(-stddev, stddev));
  }
  for(int m = 0; m  < M; m++) {
    float* W_m = W + m * 4;
    svd2x2(W_m, U, S, V);
    float p = U[0] * V[0] + U[1] * V[2];
    float q = U[0] * V[1] + U[1] * V[3];
    float r = U[2] * V[0] + U[3] * V[2];
    float s = U[2] * V[1] + U[3] * V[3];
    W_m[0] = p, W_m[1] = q, W_m[2] = r, W_m[3] = s;
    std::cout<<"Deter: "<<p * s - q * r<<std::endl;
  }
  delete [] U;
  delete [] S;
  delete [] V;
}


void cpu_permute(int M, int N, const float* PI,
                 const float* X, float* Y) {
  int MN = M * N;
  for(int  index  = 0; index < MN; index++) {
    const int n = index % N;
    const int m = index / N;
    Y[index] = X[N * m + int(PI[n])];
   }
}


void cpu_hadamard(int M, int N, float* B) {
  bool ispow2 = !(N == 0) && !(N & (N - 1));
  assert(ispow2 == true);
  for(size_t m = 0; m < M; m++) {
    float* B_m = B + m * N;
    for (size_t i = 0; i < N; i += 2) {
      size_t j = i + 1;
      float T1 = B_m[i];
      float T2 = B_m[j];
      B_m[i] = (T1 + T2);
      B_m[j] = (T1 - T2);
    }
    for(size_t bit = 2; bit < N; bit <<= 1) {
      for(size_t i = 0; i < N; i++) {
        if((bit & i) == 0) {
          size_t j =  bit | i;
          float T1 = B_m[i];
          float T2 = B_m[j];
          B_m[i] = (T1 + T2);
          B_m[j] = (T1 - T2);
        }
      }
    }
  }
}


void cpu_random_unitary_matrix(int N, float* W) {
  float *R = new float [2 * N];
  float stddev = sqrt(float(6.) / float(2 * N));
  cpu_uniform(2 * N, -stddev, stddev,  R);
  float nrm2 = cpu_dot(2 * N, R, R) / float(2);
  for(int  i = 0; i < N; i++) {
    float *W_i = W + 2 * i * N;
    for(int j = 0 ; j < N; j++) {
      int flag = int(i == j);
      W_i[2 * j] = flag;
          //-((R[2 * i] * R[2 * j] +  R[2 * i + 1] * R[2 * j + 1]) / nrm2); 
      W_i[2 * j + 1] = 0;
          //-(R[2 * i + 1] * R[2 * j] -  R[2 * i] * R[2 * j + 1]) / nrm2; 
    }
  }
  delete [] R;
}

void cpu_random_orthogonal_matrix(int N, float* W) {
  float *R = new float [ N];
  cpu_uniform(N, -1.0, 1.0,  R);
  float nrm2 = cpu_dot(N, R, R) / float(2);
  for(int  i = 0; i < N; i++) {
    float *W_i = W + i * N;
    for(int j = 0 ; j < N; j++) {
      int flag = int(i == j);
      W_i[j] = flag -((R[i] * R[j]) / nrm2); 
    }
  }
  
  delete [] R;
}


void cpu_initialize_orthogonal(int NF, int* factor_sizes,  float *W) {
  float *Wf = W;
  for(int f = 0; f < NF; f++) {
    cpu_random_orthogonal_matrix(factor_sizes[f], Wf);
    Wf = Wf + int(pow(factor_sizes[f], 2));
  }
}


float cpu_complexdet2x2(const float W_m[8], float D[2]) {
  float r1 = W_m[0] * W_m[6] - W_m[1] * W_m[7];
  float i1 = W_m[1] * W_m[6] + W_m[0] * W_m[7];
  float r2 = W_m[2] * W_m[4] - W_m[3] * W_m[5];
  float i2 = W_m[2] * W_m[5] + W_m[3] * W_m[4];
  D[0] = r1 - r2;
  D[1] = i1 - i2;
  return (pow((r1-r2), 2) + pow((i1-i2), 2));
}

void cpu_initialize_complexsvd2x2(int N, float* W) {
  int M = log2(N);
  float* U = new float[8];
  float* S = new float[2];
  float* V = new float[8];
  float stddev = sqrt(float(3.) / float(N));
  for(int i = 0; i < 8 * M; i++) {
    W[i] = float(Arena::rng_stream().uniform_rng(-stddev, stddev));
  }
  for(int m = 0; m  < M; m++) {
    float* W_m = W + m * 8;
    complexsvd2x2(W_m, U, S, V);
    matmulcomplex2x2(U, V, W_m);

    float r1 = W_m[0] * W_m[6] - W_m[1] * W_m[7];
    float i1 = W_m[1] * W_m[6] + W_m[0] * W_m[7];
    float r2 = W_m[2] * W_m[4] - W_m[3] * W_m[5];
    float i2 = W_m[2] * W_m[5] + W_m[3] * W_m[4];
    std::cout<<"Deter: "<<cpu_complexdet2x2(W_m, S)<<std::endl;
  }
  delete [] U;
  delete [] S;
  delete [] V;
}


void cpu_complex_modulus_interleaved_approxtanh(int M, int N, const float *X,
                                                const float* b, float* Y) {
  int MN = M * N;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    float* Y_m = Y + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float realb = b[2 * n];
      float imagb = b[2 * n + 1];
      realXn += realb;
      imagXn += imagb;
      float absz  = sqrt(pow(realXn, 2) + pow(imagXn, 2));
      absz = absz < 50 ? absz : 50;
      float tabsz = 2 * (1 - exp(-absz)) - 1;
      Y_m[2 * n] = realXn * tabsz;
      Y_m[2 * n + 1] = imagXn * tabsz;
    }
  }
}

void cpu_complex_modulus_interleaved_dapproxtanh(int M, int N, const float  *X,
                                                 const float* b, const float* gradY,
                                                 float* gradX, float* gradb) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    const float* gradY_m = gradY + 2 * m * N;
    float* gradX_m = gradX + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realb = b[2 * n];
      float imagb = b[2 * n + 1];
      float realbXn = X_m[2 * n] + realb;
      float imagbXn = X_m[2 * n + 1] + imagb;
      float realbXn2 = pow(realbXn, 2);
      float imagbXn2 = pow(imagbXn, 2);
      float absz  = sqrt(pow(realbXn, 2) + pow(imagbXn, 2));
      absz = absz < 50 ? absz : 50;
      float expabsz = exp(-absz);
      float tabsz = 2 * (1 - expabsz) - 1;
      const float realgradYn = gradY_m[2 * n];
      const float imaggradYn = gradY_m[2 * n + 1];
      float K1 = 2 *  expabsz / absz;
      float realdbn = tabsz + realbXn2 * K1;
      float imagdbn = realbXn * imagbXn * K1;
      float realgradbn = (realdbn * realgradYn + imagdbn * imaggradYn);
      realdbn = imagdbn;
      imagdbn = tabsz +  imagbXn2 * K1;
      float imaggradbn = (realdbn * realgradYn + imagdbn * imaggradYn);
      gradb[2 * n] += realgradbn;
      gradb[2 * n + 1] += imaggradbn;
      gradX_m[2 * n] = realgradbn;
      gradX_m[2 * n + 1] = imaggradbn;
    }
  }
}

float cpu_relative_error(int n,  const float* grad_ptr,  const float* numer_grad_ptr) {
  float rel_error = 0;
  for(int i = 0 ; i < n; i++) {
    float a =  grad_ptr[i];
    float b =  numer_grad_ptr[i];
    rel_error += (fabs(a - b))/ fmax(fabs(a), fabs(b) + 1e-5);
  }
  rel_error /= n;
  return rel_error;
}

/*
void cpu_complex_modulus_relu(int M, int N, const float *X,
                              const float* b, float* Y) {
  int MN = M * N;
  const float* realX = X;
  const float* imagX = X + MN;
  float* realY = Y;
  float* imagY = Y + MN;
  for(int m = 0; m < M; m++) {
    const float* realX_m = realX + m * N;
    const float* imagX_m = imagX + m * N;
    float* realY_m = realY + m * N;
    float* imagY_m = imagY + m * N;
    for(int n = 0; n < N; n++) {
      float realXn = realX_m[n];
      float imagXn = imagX_m[n];
      float bn = b[n];
      float absz  = sqrt(realXn * realXn + imagXn * imagXn + float(1e-7));
      realY_m[n] = 0;
      imagY_m[n] = 0;
      if(absz + bn  > 0) {
        float tabsz = (absz + bn) / absz;
        realY_m[n] = realXn * tabsz;
        imagY_m[n] = imagXn * tabsz;
      }
    }
  }
}
*/


/*
 This is a sigmoid kind of scaling mimikcked using ReLU hence it is not exactly a sigmoid
 */
void cpu_complex_modulus_interleaved_sigmoid(int M, int N, const float *X,
                                             const float* b, float* Y) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    float* Y_m = Y + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float absz  = float(1) / (epsilon + sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)));
      float bn = exp(-b[n]);
      Y_m[2 * n] = 0;
      Y_m[2 * n + 1] = 0;
      float tabsz = float(1) - bn * absz;
      if(tabsz  > 0) {
        Y_m[2 * n] = realXn * tabsz;
        Y_m[2* n + 1] = imagXn * tabsz;
      }
    }
  }
}


void cpu_complex_modulus_interleaved_dsigmoid(int M, int N, const float  *X,
                                              const float* b, const float* gradY,
                                              float* gradX, float* gradb) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    const float* gradY_m = gradY + 2 * m * N;
    float* gradX_m = gradX + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float bn = exp(-b[n]);
      float absz  = 1 / (sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)) + epsilon);
      gradX_m[2 * n] = 0;
      gradX_m[2 * n + 1] = 0;
      float tabsz = float(1) - bn * absz;
      if(tabsz > 0) {
        const float realgradYn = gradY_m[2 * n];
        const float imaggradYn = gradY_m[2 * n + 1];
        float gradbn = (realXn * realgradYn + imaggradYn * imagXn) * absz * bn;
        gradb[n] += gradbn;
        float k1 = -bn * absz;
        float abszcub = pow(absz, 3);
        float k1k2 = bn * realXn * abszcub;
        float realdXn = 1 + k1 + k1k2 * realXn;
        float imagdXn = imagXn * k1k2;
        gradX_m[2 * n] = realdXn * realgradYn + imagdXn * imaggradYn;
        k1k2 =  bn * imagXn * abszcub ;
        realdXn = realXn * k1k2;
        imagdXn = 1 + k1 + k1k2 * imagXn;
        gradX_m[2 * n + 1] =  realdXn * realgradYn + imagdXn * imaggradYn;
      }
    }
  }
}


void cpu_complex_modulus_relu(int M, int N, const float *X,
                              const float* b, float* Y) {
  int MN = M * N;
  const float* realX = X;
  const float* imagX = X + MN;
  float* realY = Y;
  float* imagY = Y + MN;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* realX_m = realX + m * N;
    const float* imagX_m = imagX + m * N;
    float* realY_m = realY + m * N;
    float* imagY_m = imagY + m * N;
    for(int n = 0; n < N; n++) {
      float realXn = realX_m[n];
      float imagXn = imagX_m[n];
      float absz  = float(1) / (epsilon + sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)));
      float bn = b[n];
      realY_m[n] = 0;
      imagY_m[n] = 0;
      float tabsz = float(1) + bn * absz;
      if(tabsz  > 0) {
        realY_m[n] = realXn * tabsz;
        imagY_m[n] = imagXn * tabsz;
      }
    }
  }
}

void cpu_complex_modulus_drelu(int M, int N, const float  *X,
                               const float* b, const float* gradY, float* gradX,
                               float* gradb) {
  int MN = M * N;
  const float* realX = X;
  const float* imagX = X + MN;
  const float* realgradY= gradY;
  const float* imaggradY = gradY + MN;
  float* realgradX = gradX;
  float* imaggradX = gradX + MN;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* realX_m = realX + m * N;
    const float* imagX_m = imagX + m * N;
    const float* realgradY_m = realgradY + m * N;
    const float* imaggradY_m = imaggradY + m * N;
    float* realgradX_m = realgradX + m * N;
    float* imaggradX_m = imaggradX + m * N;
    for(int n = 0; n < N; n++) {
      float realXn = realX_m[n];
      float imagXn = imagX_m[n];
      float bn = b[n];
      float absz  = 1 / (sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)) + epsilon);
      realgradX_m[n] = 0;
      imaggradX_m[n] = 0;
      float tabsz = float(1) + bn * absz;
      if(tabsz > 0) {
        const float realgradYn = realgradY_m[n];
        const float imaggradYn = imaggradY_m[n];
        float gradbn = (realXn * realgradYn + imaggradYn * imagXn) * absz;
        gradb[n] += gradbn;
        float k1 = bn * absz;
        float abszcub = pow(absz, 3);
        float k1k2 = bn * realXn * abszcub;
        float realdXn = 1 + k1 - k1k2 * realXn;
        float imagdXn = -imagXn * k1k2;
        realgradX_m[n] = realdXn * realgradYn + imagdXn * imaggradYn;
        k1k2 =  bn * imagXn * abszcub ;
        realdXn = -realXn * k1k2;
        imagdXn = 1 + k1 - k1k2 * imagXn;
        imaggradX_m[n] =  realdXn * realgradYn + imagdXn * imaggradYn;
      }
    }
  }
}

void cpu_complex_modulus_interleaved_relu(int M, int N, const float *X,
                                          const float* b, float* Y) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    float* Y_m = Y + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float absz  = float(1) / (epsilon + sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)));
      float bn = b[n];
      Y_m[2 * n] = 0;
      Y_m[2 * n + 1] = 0;
      float tabsz = float(1) + bn * absz;
      if(tabsz  > 0) {
        Y_m[2 * n] = realXn * tabsz;
        Y_m[2* n + 1] = imagXn * tabsz;
      }
    }
  }
}

void cpu_complex_modulus_interleaved_drelu(int M, int N, const float  *X,
                                           const float* b, const float* gradY,
                                           float* gradX, float* gradb) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    const float* gradY_m = gradY + 2 * m * N;
    float* gradX_m = gradX + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float bn = b[n];
      float absz  = 1 / (sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)) + epsilon);
      gradX_m[2 * n] = 0;
      gradX_m[2 * n + 1] = 0;
      float tabsz = float(1) + bn * absz;
      if(tabsz > 0) {
        const float realgradYn = gradY_m[2 * n];
        const float imaggradYn = gradY_m[2 * n + 1];
        float gradbn = (realXn * realgradYn + imaggradYn * imagXn) * absz;
        gradb[n] += gradbn;
        float k1 = bn * absz;
        float abszcub = pow(absz, 3);
        float k1k2 = bn * realXn * abszcub;
        float realdXn = 1 + k1 - k1k2 * realXn;
        float imagdXn = -imagXn * k1k2;
        gradX_m[2 * n] = realdXn * realgradYn + imagdXn * imaggradYn;
        k1k2 =  bn * imagXn * abszcub ;
        realdXn = -realXn * k1k2;
        imagdXn = 1 + k1 - k1k2 * imagXn;
        gradX_m[2 * n + 1] =  realdXn * realgradYn + imagdXn * imaggradYn;
      }
    }
  }
}



void cpu_complex_modulus_interleaved_tanh(int M, int N, const float *X,
                                          const float* b, float* Y) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    float* Y_m = Y + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float absz  = float(1) / (epsilon + sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)));
      float bn = b[n];
      Y_m[2 * n] = 0;
      Y_m[2 * n + 1] = 0;
      float tabsz = tanh(float(1) + bn * absz);
      Y_m[2 * n] = realXn * tabsz;
      Y_m[2* n + 1] = imagXn * tabsz;
    }
  }
}


void cpu_complex_modulus_interleaved_dtanh(int M, int N, const float  *X,
                                           const float* b, const float* gradY,
                                           float* gradX, float* gradb) {
  int MN = M * N;
  float epsilon  = 1e-5;
  for(int m = 0; m < M; m++) {
    const float* X_m = X + 2 * m * N;
    const float* gradY_m = gradY + 2 * m * N;
    float* gradX_m = gradX + 2 * m * N;
    for(int n = 0; n < N; n++) {
      float realXn = X_m[2 * n];
      float imagXn = X_m[2 * n + 1];
      float bn = b[n];
      float absz  = 1 / (sqrt(epsilon + pow(realXn, 2) + pow(imagXn, 2)) + epsilon);
      gradX_m[2 * n] = 0;
      gradX_m[2 * n + 1] = 0;
      float tabsz = tanh(float(1) + bn * absz);
      float dtabsz = float(1) -  pow(tabsz, 2);
      const float realgradYn = gradY_m[2 * n];
      const float imaggradYn = gradY_m[2 * n + 1];
      float gradbn = (realXn * realgradYn + imaggradYn * imagXn) * dtabsz * absz;
      gradb[n] += gradbn;

      float k1 = bn * absz;
      float abszcub = pow(absz, 3);
      float k1k2 = bn * realXn * abszcub;
      float realdXn = tabsz - k1k2 * realXn * dtabsz;
      float imagdXn = -imagXn * k1k2 * dtabsz;
      gradX_m[2 * n] = realdXn * realgradYn + imagdXn * imaggradYn;

      k1k2 =  bn * imagXn * abszcub ;
      realdXn = -realXn * k1k2 * dtabsz;
      imagdXn = tabsz - k1k2 * imagXn * dtabsz;
      gradX_m[2 * n + 1] =  realdXn * realgradYn + imagdXn * imaggradYn;
    }
  }
}

void cpu_uniform(int  N, const float r1, const float r2, float* X) {
  for(int i = 0; i < N; i++) {
    X[i] = float(Arena::rng_stream().uniform_rng(r1, r2));
  }
}

void cpu_gemm(const CBLAS_TRANSPOSE TransA,
              const CBLAS_TRANSPOSE TransB, int M, int N, int K,
              const float alpha, const float* A, const float* B,
              const float beta, float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

void cpu_complex_gemm(const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                      const float alpha, const float* A, const float* B,
                      const float beta, float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  float a[2] = {alpha, 0};
  float b[2] = {beta, 0};
  cblas_cgemm(CblasRowMajor, TransA, TransB, M, N, K, a, A, lda, B,
              ldb, b, C, N);
}

//The  matrix A is real
void cpu_real_complex_gemm(const CBLAS_TRANSPOSE TransA,
                           const CBLAS_TRANSPOSE TransB, int M, int N, int K,
                           const float alpha, const float* A, const float* B,
                           const float beta, float* C) {
  int D =  2 * M * K;
  float *T = new float [D];
  memset(T, 0, D * sizeof(float));
  cblas_saxpy(D / 2, 1, A, 1, T, 2);
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  float a[2] = {alpha, 0};
  float b[2] = {beta, 0};
  cblas_cgemm(CblasRowMajor, TransA, TransB, M, N, K, a, T, lda, B,
              ldb, b, C, N);
  delete [] T;
}

void cpu_gemv(const CBLAS_TRANSPOSE TransA, int M,
              int N, const float alpha, const float* A, const float* x,
              const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void cpu_axpy(int N, const float alpha, const float* X,
              float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

void cpu_axpby(int N, const float alpha, const float* X,
               const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

void cpu_copy(int N, const float* X, float* Y) {
  cblas_scopy(N, X, 1, Y, 1);
}

void cpu_scal(int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

float cpu_dot(int n, const float* x, const float* y) {
  return cblas_sdot(n, x, 1, y, 1);
}
float cpu_nrm2(int n, const float* x) {
  return cblas_snrm2(n, x, 1);
}

void  cpu_exp(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = exp(X[i]);
}

void  cpu_dexp(int N, const float *X, const float* Z,  float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = exp(X[i]) * Z[i];
}


void  cpu_tanh(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = tanh(X[i]);
}
void  cpu_sin(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = sin(X[i]);
}

void  cpu_cos(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = cos(X[i]);
}

void  cpu_dtanh(int N, const float* x, const float*  z,  float *y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i) {
    y[i] = z[i] * (float(1) - x[i] * x[i]);
  }
}

inline float sigmoid(float x) {
  return float(1) / (float(1) + exp(-x));
}

void  cpu_sigmoid(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = sigmoid(X[i]);
}

void  cpu_dsigmoid(int N, const float* x, const float*  z,  float *y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i) {
    y[i] = z[i] * (x[i] - x[i] * x[i]);
  }
}

void  cpu_log(int N, const float *X, float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = log(X[i]);
}

void  cpu_sqr(int N, const float *X, float *Y) {
/*
#pragma omp parallel num_threads(4)
      #pragma omp for
*/
  for(int i = 0; i < N; ++i)
    Y[i] = sqrt(X[i]);
}


void  cpu_pow(int N, const float *X, const float alpha,  float *Y) {
/*
#pragma omp parallel num_threads(4)
      #pragma omp for
*/
  for(int i = 0; i < N; ++i)
    Y[i] = pow(X[i], alpha);
}

void  cpu_clip(int N, const float *X, const float alpha,  float *Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i) {
    float xi = X[i];
    float fabsxi = fabs(xi);
    if(fabsxi < alpha) {
      Y[i] = xi;

    }
    else {
      Y[i] = alpha * xi / fabsxi;
    }
  }
}

void cpu_addscalar(int N, const float alpha, const float* X,
                   float* Y) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Y[i] = X[i] + alpha;
}


void  cpu_add(int N, const float *X, const float *Y,  float *Z) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Z[i] =  X[i] + Y[i];
}

void  cpu_mul(int N, const float *X, const float *Y,  float *Z) {
#pragma omp parallel num_threads(4)
      #pragma omp for
  for(int i = 0; i < N; ++i)
    Z[i] =  X[i] * Y[i];
}

void  cpu_div(int N, const float *X, const float *Y,  float *Z) {
/*
#pragma omp parallel num_threads(4)
      #pragma omp for
*/
  for(int i = 0; i < N; ++i)
    Z[i] =  X[i] / Y[i];
}


void cpu_lecun98(int N, int fan_in, float *x) {
float range = sqrt(float(3.)) / float(2 * fan_in);
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++) {
    x[i] = float(Arena::rng_stream().uniform_rng(-range, range));
  }
}

void cpu_glorot10(int fan_in, int fan_out, float *x) {
  int N = fan_in * fan_out;
  float stddev = sqrt(float(6.) / float(fan_in + fan_out));
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++) {
    x[i] = float(Arena::rng_stream().uniform_rng(-stddev, stddev));
  }
}

void cpu_initialize_glorot10(int NF, int* factor_sizes,  float *W) {
  float *Wf = W;
  float numer = 6;
  for(int f = 0; f < NF; f++) {
    float stddev = sqrt(numer / float(2 * factor_sizes[f]));
    int N = pow(factor_sizes[f], 2);
    for(int i = 0; i < N; i++) {
      Wf[i] = float(Arena::rng_stream().uniform_rng(-stddev, stddev));
    }
    Wf = Wf + N;   
  }
}


void cpu_he15(int N, int fan_in, float *x) {
  float stddev = sqrt(float(2.) / float(fan_in));
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++) {
    x[i] = float(Arena::rng_stream().gauss_rng() * stddev);
  }
}
void cpu_allzero(int N, float *x) {
  for(size_t i = 0; i < N; i++) { x[i] =  0; }
}

void cpu_fill(int N, float a, float *x) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++) { x[i] =  a; }
}

void cpu_normal(int N, const float mean, const float stddev, float *x) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++)
    x[i] = float(Arena::rng_stream().gauss_rng() * stddev + mean);
}


void cpu_chi(int N, int dof, float *x) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for(int i = 0; i < N; i++) {
    float chi2rv = 0;
    for(int j = 0; j < dof; j++) {      
      float r = float(Arena::rng_stream().gauss_rng());
      chi2rv += r * r;      
    }
    x[i] = sqrt(chi2rv);
  }
}



void cpu_rademacher(int N, float *x) {
#pragma omp parallel num_threads(4)
#pragma omp for
  for(size_t i = 0; i < N; i++) {
    x[i] = float(Arena::rng_stream().gauss_rng()) > 0 ? float(1) : float(-1);
  }
}


void cpu_permutation_array(int N, float *x) {
  for(int i = 0; i < N; i++) {
    x[i] = float(i);
  }
  for(int i = 0; i < N; i++) {
    int r =  Arena::rng_stream().randi(i, N);
    int tmp = x[r];
    x[r] = x[i];
    x[i] = tmp;
  }

  
}



float cpu_max(int N, const float* x) {
  float max = x[0];
  for(int  i = 1; i < N; i++) {
    max = x[i] > max ? x[i] : max;
  }
  return max;
}
int cpu_max_index(int N, const float* x) {
  float max = x[0];
  int index = 0;
  for(int  i = 1; i < N; i++) {
    if(x[i] > max) {
      max = x[i];
      index  = i;
    }
  }
  return index;
}
float cpu_min(int N, const float* x) {
  float min = x[0];
  for(int  i = 1; i < N; i++) {
    min = x[i] < min ? x[i] : min;
  }
  return min;
}
