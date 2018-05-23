#include "arena.h"
#include "syncedmem.h"

inline void SyncedMemory::to_cpu() {
  switch(head_){
    case UNINITIALIZED:
      CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_ * sizeof(float)));
      memset(cpu_ptr_, 0, size_* sizeof(float));
      head_ = HEAD_AT_CPU;
      break;
    case HEAD_AT_GPU:
      if(cpu_ptr_ == NULL) {
        CUDA_CHECK(cudaMallocHost(&cpu_ptr_, size_ * sizeof(float)));
      }
      CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
      head_ = SYNCED;
      break;
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

inline void SyncedMemory::to_gpu() {
  switch(head_){
    case UNINITIALIZED:
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_ * sizeof(float)));
      CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_* sizeof(float)));
      reserve_ = size_;
      head_ = HEAD_AT_GPU;
      break;
    case HEAD_AT_CPU:
      if(gpu_ptr_ == NULL) {
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_ * sizeof(float)));
      }
      CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_ * sizeof(float), cudaMemcpyHostToDevice));
      head_ = SYNCED;
      break;
    case HEAD_AT_GPU:
    case SYNCED:
      break;
  }
}

void SyncedMemory::set_size(size_t size) {
  if(reserve_ > size) {
    size_ = size;
    return;
  }
  if(cpu_ptr_ != NULL) { CUDA_CHECK(cudaFreeHost(cpu_ptr_));}
  if(gpu_ptr_ != NULL){ CUDA_CHECK(cudaFree(gpu_ptr_)); }
  cpu_ptr_ = NULL, gpu_ptr_ = NULL;
  size_ = size, reserve_ = 0;
  head_ = UNINITIALIZED;  
}

const float* SyncedMemory::cpu_data() {
  to_cpu();
  return (const float*)cpu_ptr_;
}

const float* SyncedMemory::gpu_data() {
  to_gpu();
  return (const float*)gpu_ptr_;
}

float* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

float* SyncedMemory::mutable_gpu_data() {
  to_gpu();
  head_ =  HEAD_AT_GPU;
  return gpu_ptr_;
}
