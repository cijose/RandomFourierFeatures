#ifndef _RANDOM_LAYER_H_
#define _RANDOM_LAYER_H_
#include "arena.h"
#include "syncedmem.h"
#include <vector>
#include <cassert>

using namespace std;

/*
 output_dim_ =   num_neurons_ * input_dim_
*/

class RandomLayer {
 protected:
  int num_blocks_;
  int max_blocks_;
  int input_dim_;
  int output_dim_;
  int batch_size_;
  float sigma_;
  float alpha_;
  std::vector<SyncedMemory*> random_vec_;
  SyncedMemory* bias_;
  SyncedMemory* output_;
  virtual void reset_cpu(float stddev) = 0;
  virtual void reset_gpu(float stddev) = 0;
  virtual void forward_cpu(SyncedMemory* input, int batch_size) = 0;
  virtual void forward_gpu(SyncedMemory* input, int batch_size) = 0;
 public:
  RandomLayer() {}
  virtual ~RandomLayer() {}
  inline int  input_dim(){ return input_dim_; }
  inline int  hidden_dim(){ return output_dim_; }
  inline int  num_blocks(){ return num_blocks_; }
  inline int  batch_size(){ return batch_size_; }
  inline void set_num_blocks(int num_blocks) {
    assert(num_blocks < max_blocks_);
    num_blocks_ = num_blocks;
    output_dim_ = num_blocks_ * input_dim_;
  }
  inline void reset(float stddev) {
    switch (Arena::device()) {
      case Arena::CPU:
        reset_cpu(stddev);
        break;
      case Arena::GPU:
        reset_gpu(stddev);
        break;
      default:
        LOG(FATAL) << "Unknown  device.";
    }
  }
  inline void forward(SyncedMemory* input, int batch_size) {
    switch (Arena::device()) {
      case Arena::CPU:
        forward_cpu(input, batch_size);
        break;
      case Arena::GPU:
        forward_gpu(input, batch_size);
        break;
      default:
        LOG(FATAL) << "Unknown  device.";
    }
  }
  vector<SyncedMemory* >& random_weights() { return random_vec_; }
  SyncedMemory* output() { return output_; }  
};

class RechtRahimiRandomLayer: public RandomLayer {
  SyncedMemory* weight_;
  SyncedMemory* bias_;
  SyncedMemory* bias_multiplier_;
  SyncedMemory* buffer_;
 public:
  explicit RechtRahimiRandomLayer(int input_dim, int max_blocks, float sigma);
  void reset_cpu(float stddev);
  void reset_gpu(float stddev);
  void forward_cpu(SyncedMemory* input, int batch_size);
  void forward_gpu(SyncedMemory* input, int batch_size);
  
  virtual ~RechtRahimiRandomLayer() {
    delete weight_;
    delete bias_;
    delete bias_multiplier_;
    delete output_;
    delete buffer_;
  }
};
/*
 SHGPIHB
*/

class FastfoodRandomLayer: public RandomLayer {
  SyncedMemory* scaler_;
  SyncedMemory* gaussian_;
  SyncedMemory* permutation_;
  SyncedMemory* rademacher_;
  SyncedMemory* bias_;
  SyncedMemory* buffer_;
  SyncedMemory* bias_multiplier_;
 public:
  explicit FastfoodRandomLayer(int input_dim, int max_blocks, float sigma);
  void reset_cpu(float stddev);
  void reset_gpu(float stddev);
  void forward_cpu(SyncedMemory* input,  int batch_size);
  void forward_gpu(SyncedMemory* input, int batch_size);
  virtual ~FastfoodRandomLayer() {
    delete scaler_;
    delete gaussian_;
    delete permutation_;
    delete rademacher_;
    delete bias_;
    delete bias_multiplier_;
    delete output_;
    delete buffer_;
  }
};

class OrthogonalRandomLayer: public RandomLayer {
  SyncedMemory* scaler_;
  SyncedMemory* weight_;
  SyncedMemory* bias_;
  SyncedMemory* bias_multiplier_;
  SyncedMemory* buffer_;

 public:
  explicit OrthogonalRandomLayer(int input_dim, int max_blocks, float sigma);
  void reset_cpu(float stddev);
  void reset_gpu(float stddev);
  void forward_cpu(SyncedMemory* input,  int batch_size);
  void forward_gpu(SyncedMemory* input,  int batch_size);
  virtual ~OrthogonalRandomLayer() {
    delete scaler_;
    delete weight_;
    delete bias_;
    delete bias_multiplier_;
    delete output_;
    delete buffer_;
  }
};

class StructuredOrthogonalRandomLayer: public RandomLayer {
  SyncedMemory* rademacher1_;
  SyncedMemory* rademacher2_;
  SyncedMemory* rademacher3_;
  SyncedMemory* bias_;
  SyncedMemory* buffer_;
  SyncedMemory* bias_multiplier_;
 public:
  explicit StructuredOrthogonalRandomLayer(int input_dim, int max_blocks, float sigma);
  void reset_cpu(float stddev);
  void reset_gpu(float stddev);
  void forward_cpu(SyncedMemory* input, int batch_size);
  void forward_gpu(SyncedMemory* input, int batch_size);
  virtual ~StructuredOrthogonalRandomLayer() {
    delete rademacher1_;
    delete rademacher2_;
    delete rademacher3_;
    delete bias_;
    delete bias_multiplier_;
    delete output_;
    delete buffer_;
  }
};

#endif
