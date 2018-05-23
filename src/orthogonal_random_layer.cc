#include "random_layer.h"
#include "math_functions.h"

OrthogonalRandomLayer::OrthogonalRandomLayer(int input_dim, int max_blocks, float sigma) {
  sigma_ = sigma;
  alpha_ = float(1)  / (M_PI * sigma_ * sqrt(2.0));
  input_dim_ = input_dim;
  max_blocks_ = max_blocks;
  num_blocks_ = max_blocks;
  output_dim_ = input_dim * num_blocks_;
  
  scaler_ =  new SyncedMemory(output_dim_);
  weight_ =  new SyncedMemory(input_dim_ * output_dim_);
  
  bias_ = new SyncedMemory(output_dim_);
  random_vec_.push_back(scaler_);
  random_vec_.push_back(weight_);
  random_vec_.push_back(bias_);
  output_ = new SyncedMemory();
  buffer_ = new SyncedMemory();
  bias_multiplier_ = new SyncedMemory();
  if(Arena::device() == Arena::CPU) {
    cpu_chi(scaler_->size(), input_dim_, scaler_->mutable_cpu_data());
    cpu_scal(scaler_->size(), float(1) / sigma_, scaler_->mutable_cpu_data());
    for(int i = 0; i < max_blocks_; i++){
      gpu_random_orthogonal_matrix(input_dim_, weight_->mutable_gpu_data() + i * input_dim_ * input_dim_);
    }
    cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
  }
  else {
    gpu_chi(scaler_->size(), input_dim_, scaler_->mutable_gpu_data());
    gpu_scal(scaler_->size(), float(1) / sigma_, scaler_->mutable_gpu_data());
    for(int i = 0; i < max_blocks_; i++){
      gpu_random_orthogonal_matrix(input_dim_, weight_->mutable_gpu_data() + i * input_dim_ * input_dim_);
    }
    gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
  }
}

void OrthogonalRandomLayer::reset_cpu(float sigma) {
  sigma_ = sigma;
  cpu_chi(scaler_->size(), input_dim_, scaler_->mutable_cpu_data());
  cpu_scal(scaler_->size(), float(1) / sigma_, scaler_->mutable_cpu_data());
  for(int i = 0; i < max_blocks_; i++){
    gpu_random_orthogonal_matrix(input_dim_, weight_->mutable_gpu_data() + i * input_dim_ * input_dim_);
  }
  cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
}

void OrthogonalRandomLayer::reset_gpu(float sigma) {
  sigma_ = sigma;
  gpu_chi(scaler_->size(), input_dim_, scaler_->mutable_gpu_data());
  gpu_scal(scaler_->size(), float(1) / sigma_, scaler_->mutable_gpu_data());
  for(int i = 0; i < max_blocks_; i++){
    gpu_random_orthogonal_matrix(input_dim_, weight_->mutable_gpu_data() + i * input_dim_ * input_dim_);
  }
  gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
}

void print_matrix(int N, const float *W) {
  for(int  i =0; i < N * N; i++) {
    cout<<W[i]<<" ";
    if((i + 1) % N == 0) {
      cout<<endl;
    }
  }
}

void OrthogonalRandomLayer::forward_cpu(SyncedMemory* x, int batch_size) {
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
    buffer_ = new SyncedMemory( bstod);
    cpu_allzero(buffer_->size(), buffer_->mutable_cpu_data());
  }
  batch_size_ = batch_size;
  const float* M = bias_multiplier_->cpu_data(); 
  for(int i = 0; i < num_blocks_; i++) {
    const float* Wx_ptr_i = weight_->cpu_data() + i * idtid;
    float* buffer_ptr_i =  buffer_->mutable_cpu_data() + i * bstid;
    const float* S_i = scaler_->cpu_data() + i * input_dim_;
    cpu_kronecker_forward(batch_size_, input_dim_, x->cpu_data(),
                          1, &input_dim_, false,
                          Wx_ptr_i,  buffer_ptr_i);
    cpu_diag_gemm(batch_size_, input_dim_, float(1), buffer_ptr_i,
                  S_i, float(0), buffer_ptr_i);
  }
  cpu_rearrange(batch_size_, input_dim_, num_blocks_,
                buffer_->cpu_data(), output_->mutable_cpu_data());
  cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->cpu_data(), (float)1., output_->mutable_cpu_data());
  cpu_sin(bstod, output_->cpu_data(), output_->mutable_cpu_data() + bstod);  
  cpu_cos(bstod, output_->cpu_data() , output_->mutable_cpu_data());
  cpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_cpu_data());
}

void OrthogonalRandomLayer::forward_gpu(SyncedMemory* x, int batch_size) {
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
  const float* M = bias_multiplier_->gpu_data(); 
  for(int i = 0; i < num_blocks_; i++) {
    const float* Wx_ptr_i = weight_->gpu_data() + i * idtid;
    float* buffer_ptr_i =  buffer_->mutable_gpu_data() + i * bstid;
    const float* S_i = scaler_->gpu_data() + i * input_dim_;

    gpu_kronecker_forward(batch_size_, input_dim_, x->gpu_data(),
                          1, &input_dim_, false,
                          Wx_ptr_i,  buffer_ptr_i);
    gpu_diag_gemm(batch_size_, input_dim_, float(1), buffer_ptr_i,
                  S_i, float(0), buffer_ptr_i);
  }
  gpu_rearrange(batch_size_, input_dim_, num_blocks_,
                buffer_->gpu_data(), output_->mutable_gpu_data());
  gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1., M, bias_->gpu_data(), (float)1., output_->mutable_gpu_data());
  gpu_cos(bstod, output_->gpu_data() , output_->mutable_gpu_data());
  gpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_gpu_data());
}








