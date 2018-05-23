#include "random_layer.h"
#include "math_functions.h"

void cpu_sorf_transform(int batch_size, int input_dim, int num_blocks,
                        float sigma, const float *X, const float *D1,
                        const float *D2, const float* D3, float* Y) {
  for(int  n =0; n < num_blocks; n++) {
    const float* D1_n = D1 + n * input_dim;
    const float* D2_n = D2 + n * input_dim;
    const float* D3_n = D3 + n * input_dim;
    float *Y_n = Y + n * input_dim * batch_size;
    cpu_diag_gemm(batch_size, input_dim, float(1), X,
                  D3_n, float(0), Y_n);
    cpu_hadamard(batch_size, input_dim, Y_n);
    cpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  D2_n, float(0), Y_n);
    cpu_hadamard(batch_size, input_dim, Y_n);
    cpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  D1_n, float(0), Y_n);
    cpu_hadamard(batch_size, input_dim, Y_n);
  }
  int output_dim = input_dim * num_blocks;
  cpu_scal(batch_size * output_dim, float(1) / (sigma  * (input_dim)), Y);
}


void gpu_sorf_transform(int batch_size, int input_dim, int num_blocks,
                        float sigma, const float *X, const float *D1,
                        const float *D2, const float* D3, float* Y) {
  for(int  n =0; n < num_blocks; n++) {
    const float* D1_n = D1 + n * input_dim;
    const float* D2_n = D2 + n * input_dim;
    const float* D3_n = D3 + n * input_dim;
    float *Y_n = Y + n * input_dim * batch_size;
    gpu_diag_gemm(batch_size, input_dim, float(1), X,
                  D3_n, float(0), Y_n);
    gpu_hadamard(batch_size, input_dim, Y_n);
    gpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  D2_n, float(0), Y_n);
    gpu_hadamard(batch_size, input_dim, Y_n);
    gpu_diag_gemm(batch_size, input_dim, float(1), Y_n,
                  D1_n, float(0), Y_n);
    gpu_hadamard(batch_size, input_dim, Y_n);
  }
  int output_dim = input_dim * num_blocks;
  gpu_scal(batch_size * output_dim, float(1) / (sigma  * (input_dim)), Y);
}

StructuredOrthogonalRandomLayer::StructuredOrthogonalRandomLayer(int input_dim, int max_blocks, float sigma) {
  sigma_ = sigma;
  alpha_ = float(1)  / (M_PI * sigma_ * sqrt(2.0));
  input_dim_ = input_dim;
  max_blocks_= max_blocks;
  num_blocks_ = max_blocks;
  output_dim_ = input_dim * max_blocks_;
  
  rademacher1_ =  new SyncedMemory(output_dim_);
  rademacher2_ =  new SyncedMemory(output_dim_);
  rademacher3_ =  new SyncedMemory(output_dim_);
  output_ = new SyncedMemory();
  buffer_ = new SyncedMemory();
  bias_multiplier_ = new SyncedMemory();

  bias_ = new SyncedMemory(output_dim_);
  random_vec_.push_back(rademacher1_);
  random_vec_.push_back(rademacher2_);
  random_vec_.push_back(rademacher3_);
  random_vec_.push_back(bias_);

  if(Arena::device() == Arena::CPU) {
    cpu_rademacher(rademacher1_->size(), rademacher1_->mutable_cpu_data());
    cpu_rademacher(rademacher2_->size(), rademacher2_->mutable_cpu_data());
    cpu_rademacher(rademacher3_->size(), rademacher3_->mutable_cpu_data());
    cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
  }
  else {
    gpu_rademacher(rademacher1_->size(), rademacher1_->mutable_gpu_data());
    gpu_rademacher(rademacher2_->size(), rademacher2_->mutable_gpu_data());
    gpu_rademacher(rademacher3_->size(), rademacher3_->mutable_gpu_data());
    gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
  }
}

void StructuredOrthogonalRandomLayer::reset_cpu(float sigma) {
  sigma_ = sigma;
  cpu_rademacher(rademacher1_->size(), rademacher1_->mutable_cpu_data());
  cpu_rademacher(rademacher2_->size(), rademacher2_->mutable_cpu_data());
  cpu_rademacher(rademacher3_->size(), rademacher3_->mutable_cpu_data());
  cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
  
}

void StructuredOrthogonalRandomLayer::reset_gpu(float sigma) {
  sigma_ = sigma;
  gpu_rademacher(rademacher1_->size(), rademacher1_->mutable_gpu_data());
  gpu_rademacher(rademacher2_->size(), rademacher2_->mutable_gpu_data());
  gpu_rademacher(rademacher3_->size(), rademacher3_->mutable_gpu_data());
  gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
}

void StructuredOrthogonalRandomLayer::forward_cpu(SyncedMemory* x, int batch_size) {
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
  const float* D1 = rademacher1_->cpu_data();
  const float* D2 = rademacher2_->cpu_data();
  const float* D3 = rademacher3_->cpu_data();
  const float* M = bias_multiplier_->cpu_data(); 
  float* T = buffer_->mutable_cpu_data();
  float* Y = output_->mutable_cpu_data();
  cpu_sorf_transform(batch_size_, input_dim_, num_blocks_,
                     sigma_, X, D1, D2,  D3, T);
  cpu_rearrange(batch_size_, input_dim_, num_blocks_, T, Y);
  cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->cpu_data(), (float)1., Y);
  cpu_cos(bstod, output_->cpu_data() , output_->mutable_cpu_data());
  cpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_cpu_data());
}

void StructuredOrthogonalRandomLayer::forward_gpu(SyncedMemory* x, int batch_size) {
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
  const float* D1 = rademacher1_->gpu_data();
  const float* D2 = rademacher2_->gpu_data();
  const float* D3 = rademacher3_->gpu_data();
  const float* M = bias_multiplier_->gpu_data(); 
  float* T = buffer_->mutable_gpu_data();
  float* Y = output_->mutable_gpu_data();
  gpu_sorf_transform(batch_size_, input_dim_, num_blocks_,
                     sigma_, X, D1, D2,  D3, T);
  gpu_rearrange(batch_size_, input_dim_, num_blocks_, T, Y);
  gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->gpu_data(), (float)1., Y);
  gpu_cos(bstod, output_->gpu_data() , output_->mutable_gpu_data());
  gpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_gpu_data());
}






