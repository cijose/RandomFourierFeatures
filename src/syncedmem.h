#ifndef _SYNCEDMEM_H_
#define _SYNCEDMEM_H_

class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), reserve_(0), head_(UNINITIALIZED) {}
  explicit SyncedMemory(size_t size)
      :cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), reserve_(0), head_(UNINITIALIZED){}
  ~SyncedMemory() {
    if(cpu_ptr_ != NULL) { CUDA_CHECK(cudaFreeHost(cpu_ptr_)); }
    if(gpu_ptr_ != NULL){ CUDA_CHECK(cudaFree(gpu_ptr_)); }
  }
  enum SyncedHead {UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED};
  const float* cpu_data();
  const  float* gpu_data();
  float* mutable_cpu_data();
  float* mutable_gpu_data();
  SyncedHead head() { return head_; }
  void set_size(size_t size);
  size_t size() { return size_; }
 private:
  void to_gpu();
  void to_cpu();
  float* cpu_ptr_;
  float* gpu_ptr_;
  size_t size_;
  size_t reserve_;
  SyncedHead head_;
  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};


#endif
