#ifndef _ARENA_H_
#define _ARENA_H_

/*
Stolen from caffe
*/

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#include <glog/logging.h>
#include "random.h"

#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>

#define CUDA_CHECK(condition) CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition) CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define CURAND_CHECK(condition) CHECK_EQ((condition), CURAND_STATUS_SUCCESS)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define DISABLE_COPY_AND_ASSIGN(classname)\
private:\
 classname(const classname&);\
 classname& operator=(const classname&)


using std::shared_ptr;
class ArenaError : public std::exception {
 public:
  explicit ArenaError(const std::string& msg) : msg_(msg){}
  ~ArenaError() throw() {}
  const char* what() const throw() { return msg_.c_str(); }
 private:
  std::string msg_;
  DISABLE_COPY_AND_ASSIGN(ArenaError);
};

#if __CUDA_ARCH__ >= 200
const size_t CUDA_NUM_THREADS = 1024;
#else
const size_t CUDA_NUM_THREADS = 512;
#endif

inline size_t CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

class cervnet_error : public std::exception {
  public:
  explicit cervnet_error(const std::string& msg) : msg_(msg){}
  ~cervnet_error() throw() {}
  const char* what() const throw() { return msg_.c_str(); }
  private:
  std::string msg_;
};

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas.
class Arena {
 private:
  // The private constructor to avoid duplicate instantiation.
  Arena();
 protected:
  static shared_ptr<Arena> singleton_;
 public:
  ~Arena();
  inline static Arena& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Arena());
    }
    return *singleton_;
  }
  enum Device{CPU, GPU};
  enum Phase { TRAIN, TEST };
  // The getters for the variables.
  // Returns the cublas handle.
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  // Returns the curand generator.
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
  //Getter for Prand random numbers
  inline static Prand& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new Prand());
    }
    return *(Get().random_generator_);
  }
  // Returns the mode: running on CPU or GPU.
  inline static Device device() { return Get().device_; }
  // Returns the phase: TRAIN or TEST.
  inline static Phase phase() { return Get().phase_; }
  // The setters for the variables
  // Sets the mode.
  inline static void set_device(Device device) {Get().device_ = device; }
  // Sets the phase.
  inline static void set_phase(Phase phase) { Get().phase_ = phase; }
 protected:
  Device device_;
  Phase phase_;
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
  shared_ptr<Prand> random_generator_;
  DISABLE_COPY_AND_ASSIGN(Arena);
};

#endif
