#include "random_layer.h"
#include "math_functions.h"
#include "kronecker_utils.h"

RechtRahimiRandomLayer::RechtRahimiRandomLayer(int input_dim, int max_blocks, float sigma) {
  sigma_ = sigma;
  input_dim_ = input_dim;
  max_blocks_ = max_blocks;
  num_blocks_ = max_blocks; 
  output_dim_ = input_dim * num_blocks_;
  int m = input_dim_ * output_dim_;
  weight_ = new SyncedMemory(m);
  bias_ = new SyncedMemory(output_dim_);
  random_vec_.push_back(weight_);
  random_vec_.push_back(bias_);
  output_ = new SyncedMemory();
  buffer_ = new SyncedMemory();
  bias_multiplier_ = new SyncedMemory();
  cout<<"PI "<<M_PI<<endl;
  if(Arena::device() == Arena::CPU) {
    cpu_normal(weight_->size(), 0, 1.0 / sigma ,  weight_->mutable_cpu_data());
    cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
  }
  else {
    gpu_normal(weight_->size(), 0, 1.0 / sigma,  weight_->mutable_gpu_data());
    gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
  }
}

void RechtRahimiRandomLayer::reset_cpu(float sigma) {
  sigma_ = sigma;
  cpu_normal(weight_->size(), 0, 1.0 / sigma,  weight_->mutable_cpu_data());
  cpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_cpu_data());
}

void RechtRahimiRandomLayer::reset_gpu(float sigma) {
  sigma_ = sigma;
  gpu_normal(weight_->size(), 0, 1.0 / sigma,  weight_->mutable_gpu_data());
  gpu_uniform(bias_->size(), 0, float(2.0 * M_PI), bias_->mutable_gpu_data());
}

void RechtRahimiRandomLayer::forward_cpu(SyncedMemory* x, int batch_size) {
  const int bstid = batch_size * input_dim_;
  const int bstod = batch_size * output_dim_;
  const int idtid = input_dim_ * input_dim_;
  CHECK_EQ(x->size(), bstid) << "Input size should be multiple of batch size times input dim";
  if(bias_multiplier_->size() != batch_size) {
    delete bias_multiplier_;
    bias_multiplier_ = new SyncedMemory(batch_size);
    cpu_fill(bias_multiplier_->size(), float(1), bias_multiplier_->mutable_cpu_data());
  }
  if(output_->size() != bstod) {
    delete output_;
    output_ = new SyncedMemory( bstod);
    cpu_allzero(output_->size(), output_->mutable_cpu_data());
  }
  if(buffer_->size() !=  bstod) {
    delete buffer_;
    buffer_ = new SyncedMemory(bstod);
    cpu_allzero(buffer_->size(), buffer_->mutable_cpu_data());
  }
  batch_size_ = batch_size;
  for(int i = 0; i < num_blocks_; i++) {
    const float* Wx_ptr_i = weight_->cpu_data() + i * idtid;
    float* buffer_ptr_i =  buffer_->mutable_cpu_data() + i * bstid;
    cpu_kronecker_forward(batch_size_, input_dim_, x->cpu_data(),
                          1, &input_dim_, false,
                          Wx_ptr_i,  buffer_ptr_i);
  }
  cpu_rearrange(batch_size_, input_dim_, num_blocks_,
                buffer_->cpu_data(), output_->mutable_cpu_data());
  cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1.,  bias_multiplier_->cpu_data(),
           bias_->cpu_data(), (float)1., output_->mutable_cpu_data());
  cpu_cos(bstod, output_->cpu_data(), output_->mutable_cpu_data());
  cpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_cpu_data());
}

void RechtRahimiRandomLayer::forward_gpu(SyncedMemory* x, int batch_size) {
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
  for(int i = 0; i < num_blocks_; i++) {
    const float* Wx_ptr_i = weight_->gpu_data() + i * idtid;
    float* buffer_ptr_i =  buffer_->mutable_gpu_data() + i * bstid;
    gpu_kronecker_forward(batch_size_, input_dim_, x->gpu_data(),
                          1, &input_dim_, false,
                          Wx_ptr_i,  buffer_ptr_i);
  }
  gpu_rearrange(batch_size_, input_dim_, num_blocks_,
                buffer_->gpu_data(), output_->mutable_gpu_data());
  gpu_gemm(CblasNoTrans, CblasTrans, batch_size_, output_dim_,
           1, (float)1.,  bias_multiplier_->gpu_data(),
           bias_->gpu_data(), (float)1., output_->mutable_gpu_data());
  gpu_cos(bstod, output_->gpu_data() , output_->mutable_gpu_data());
  gpu_scal(output_->size(), sqrt(float(2) / float(output_dim_)), output_->mutable_gpu_data());
}




