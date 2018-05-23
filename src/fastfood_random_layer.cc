#include "random_layer.h"
#include "math_functions.h"

void cpu_fastfood_transform(int batch_size, int input_dim, int num_blocks,
                            float sigma, const float *X, const float *S,
                            const float *G, const float* PI, const float* B,
                            float *T, float* Y) {
  for(int  n =0; n < num_blocks; n++) {
    const float* S_n = S + n * input_dim;
    const float* G_n = G + n * input_dim;
    const float* PI_n = PI + n * input_dim;
    const float* B_n = B + n * input_dim;
    float *Y_n = Y + n * input_dim * batch_size;
    cpu_diag_gemm(batch_size, input_dim, float(1), X,
                  B_n, float(0), T);
    cpu_hadamard(batch_size, input_dim, T);
    cpu_permute(batch_size, input_dim, PI_n, T, Y_n);
    cpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  G_n, float(0), Y_n);
    cpu_hadamard(batch_size, input_dim, Y_n);
    cpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  S_n, float(0), Y_n);
        
  }
  int output_dim = input_dim * num_blocks;
  cpu_scal(batch_size * output_dim, float(1) / (sigma  * (input_dim)), Y);
}

void gpu_fastfood_transform(int batch_size, int input_dim, int num_blocks,
                            float sigma, const float *X, const float *S,
                            const float *G, const float* PI, const float* B,
                            float *T, float* Y) {
  for(int  n =0; n < num_blocks; n++) {
    const float* S_n = S + n * input_dim;
    const float* G_n = G + n * input_dim;
    const float* PI_n = PI + n * input_dim;
    const float* B_n = B + n * input_dim;
    float *Y_n = Y + n * input_dim * batch_size;
    gpu_diag_gemm(batch_size, input_dim, float(1), X,
                  B_n, float(0), T);
    gpu_hadamard(batch_size, input_dim, T);
    gpu_permute(batch_size, input_dim, PI_n, T, Y_n);
    gpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  G_n, float(0), Y_n);
    gpu_hadamard(batch_size, input_dim, Y_n);
    gpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  S_n, float(0), Y_n);

  }
  int output_dim = input_dim * num_blocks;
  gpu_scal(batch_size * output_dim, float(1) / (sigma * (input_dim)), Y);
}

FastfoodRandomLayer::FastfoodRandomLayer(int input_dim, int max_blocks, float sigma) {
  sigma_ = sigma;
  alpha_ = float(1)  / (M_PI * sigma_ * sqrt(2.0));
  input_dim_ = input_dim;
  max_blocks_= max_blocks;
  num_blocks_ = max_blocks;
  output_dim_ = input_dim * max_blocks_;
  
  scaler_ =  new SyncedMemory(output_dim_);
  gaussian_ =  new SyncedMemory(output_dim_);
  permutation_ =  new SyncedMemory(output_dim_);
  rademacher_ =  new SyncedMemory(output_dim_);
  output_ = new SyncedMemory();
  buffer_ = new SyncedMemory();
  bias_multiplier_ = new SyncedMemory();

  bias_ = new SyncedMemory(output_dim_);
  random_vec_.push_back(scaler_);
  random_vec_.push_back(gaussian_);
  random_vec_.push_back(permutation_);
  random_vec_.push_back(rademacher_);
  random_vec_.push_back(bias_);

  if(Arena::device() == Arena::CPU) {
    cpu_chi(scaler_->size(), input_dim_, scaler_->mutable_cpu_data());
    cpu_normal(gaussian_->size(), 0, 1, gaussian_->mutable_cpu_data());
    for(int  n = 0; n < max_blocks_; n++) {
      cpu_permutation_array(input_dim_, permutation_->mutable_cpu_data() + n * input_dim_);
    }
    cpu_rademacher(rademacher_->size(), rademacher_->mutable_cpu_data());
    cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
  }
  else {
    gpu_chi(scaler_->size(), input_dim_, scaler_->mutable_gpu_data());
    gpu_normal(gaussian_->size(), 0, 1, gaussian_->mutable_gpu_data());
    for(int  n = 0; n < max_blocks_; n++) {
      gpu_permutation_array(input_dim_, permutation_->mutable_gpu_data() + n * input_dim_);
    }
    gpu_rademacher(rademacher_->size(), rademacher_->mutable_gpu_data());
    gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
  }
}

void FastfoodRandomLayer::reset_cpu(float sigma) {
  sigma_ = sigma;
  cpu_chi(scaler_->size(), input_dim_, scaler_->mutable_cpu_data());
  cpu_normal(gaussian_->size(), 0, 1, gaussian_->mutable_cpu_data());
  for(int  n = 0; n < max_blocks_; n++) {
    cpu_permutation_array(input_dim_, permutation_->mutable_cpu_data() + n * input_dim_);
  }
  cpu_rademacher(rademacher_->size(), rademacher_->mutable_cpu_data());
  cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
}

void FastfoodRandomLayer::reset_gpu(float sigma) {
  sigma_ = sigma;
  gpu_chi(scaler_->size(), input_dim_, scaler_->mutable_gpu_data());
  gpu_normal(gaussian_->size(), 0, 1, gaussian_->mutable_gpu_data());
  for(int  n = 0; n < max_blocks_; n++) {
    gpu_permutation_array(input_dim_, permutation_->mutable_gpu_data() + n * input_dim_);
  }
  gpu_rademacher(rademacher_->size(), rademacher_->mutable_gpu_data());
  gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
}

void FastfoodRandomLayer::forward_cpu(SyncedMemory* x, int batch_size) {
  const int bstid = batch_size * input_dim_;
  const int bstod = batch_size * output_dim_;
  const int idtid = input_dim_ * input_dim_;
  CHECK_EQ(x->size(), bstid) << "Input size should be multiple of batch size times input dim";
  if(bias_multiplier_->size() != batch_size) {
    delete bias_multiplier_;
    bias_multiplier_ = new SyncedMemory(batch_size);
    cpu_fill(bias_multiplier_->size(), float(1), bias_multiplier_->mutable_cpu_data());
  }
  if(output_->size() !=  bstod) {
    delete output_;
    output_ = new SyncedMemory(bstod);
    cpu_allzero(output_->size(), output_->mutable_cpu_data());
  }
  if(buffer_->size() !=  bstod) {
    delete buffer_;
    buffer_ = new SyncedMemory(bstod);
    cpu_allzero(buffer_->size(), buffer_->mutable_cpu_data());
  }
  batch_size_ = batch_size;
  const float* X = x->cpu_data();
  const float* S = scaler_->cpu_data();
  const float* G = gaussian_->cpu_data();
  const float* PI = permutation_->cpu_data();
  const float* B = rademacher_->cpu_data();
  const float* M = bias_multiplier_->cpu_data(); 
  float* T = output_->mutable_cpu_data();
  float* Y = buffer_->mutable_cpu_data();
  cpu_fastfood_transform(batch_size_, input_dim_, num_blocks_,
                         sigma_, X, S, G, PI, B, T, Y);
  cpu_rearrange(batch_size_, input_dim_, num_blocks_, Y, T);
  cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->cpu_data(), (float)1., T);
  cpu_cos(bstod, output_->cpu_data() , output_->mutable_cpu_data());
  cpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_cpu_data());
}

void FastfoodRandomLayer::forward_gpu(SyncedMemory* x, int batch_size) {
  const int bstid = batch_size * input_dim_;
  const int bstod = batch_size * output_dim_;
  const int idtid = input_dim_ * input_dim_;
  CHECK_EQ(x->size(), bstid) << "Input size should be multiple of batch size times input dim";
  if(bias_multiplier_->size() != batch_size) {
    delete bias_multiplier_;
    bias_multiplier_ = new SyncedMemory(batch_size);
    gpu_fill(bias_multiplier_->size(), float(1), bias_multiplier_->mutable_gpu_data());
  }
  if(output_->size() !=  bstod) {
    delete output_;
    output_ = new SyncedMemory(bstod);
    gpu_allzero(output_->size(), output_->mutable_gpu_data());
  }
  if(buffer_->size() !=  bstod) {
    delete buffer_;
    buffer_ = new SyncedMemory(bstod);
    gpu_allzero(buffer_->size(), buffer_->mutable_gpu_data());
  }
  batch_size_ = batch_size;
  const float* X = x->gpu_data();
  const float* S = scaler_->gpu_data();
  const float* G = gaussian_->gpu_data();
  const float* PI = permutation_->gpu_data();
  const float* B = rademacher_->gpu_data();
  const float* M = bias_multiplier_->gpu_data(); 
  float* T = output_->mutable_gpu_data();
  float* Y = buffer_->mutable_gpu_data();
  gpu_fastfood_transform(batch_size_, input_dim_, num_blocks_,
                         sigma_, X, S, G, PI, B,  T, Y); 
  gpu_rearrange(batch_size_, input_dim_, num_blocks_, Y, T);
  gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->gpu_data(), (float)1., T);
  gpu_cos(bstod, output_->gpu_data() , output_->mutable_gpu_data());
  gpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_gpu_data());
}






