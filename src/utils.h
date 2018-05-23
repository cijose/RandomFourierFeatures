#ifndef _UTILS_H_
#define _UTILS_H_

#include <fcntl.h>
#include <unistd.h>

/*
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
*/

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>


#include <sys/time.h>
class Timer {
  timeval timer[2];
 public:
  timeval startTimer(void) {
    gettimeofday(&this->timer[0], NULL);
    return this->timer[0];
  }
  timeval stopTimer(void) {
    gettimeofday(&this->timer[1], NULL);
    return this->timer[1];
  }
  double elapsedTime(void){
    double secs(this->timer[1].tv_sec - this->timer[0].tv_sec);
    double usecs(this->timer[1].tv_usec - this->timer[0].tv_usec);
    if(usecs < 0) {
      --secs;
      usecs += 1000000;
    }
    return (secs * 1000 + usecs / 1000.0);
  }
  double elapsedTimeSeconds(void){
    double secs(this->timer[1].tv_sec - this->timer[0].tv_sec);
    double usecs(this->timer[1].tv_usec - this->timer[0].tv_usec);
    if(usecs < 0) {
      --secs;
      usecs += 1000000;
    }
    return (secs  + usecs * 1e-6);
  }
};

enum LayerType {RFF=0, FASTFOOD=1, ORF=2, SORF=3, KRF=4, AFFINE=5, LOGSOFTMAX=6, NONE=7};
enum ScalerType {NORMAL=0, MATERN=1};
enum CriterionType {NLL=0, MSE=1, NLLMSE=2};

enum OptimType {SG=0, RMS=1, ADAM=2};


struct CriterionDefinition {
  CriterionType type_;
  int num_states_;
  int batch_size_;
  int dimension_;
  bool recompute_backward_;
  CriterionDefinition(): type_(NLL), num_states_(0), batch_size_(0),  dimension_(0), recompute_backward_(true){}
};

struct LayerDefinition {
  LayerType type_;
  int num_states_;
  int batch_size_;
  int hidden_dim_;
  int input_dim_;
  int num_classes_;
  int num_factors_;
  int num_stacks_;
  bool recompute_backward_;
  bool remember_states_;
  LayerDefinition(): type_(NONE), num_states_(0), batch_size_(0), hidden_dim_(0),
                     input_dim_(0), num_classes_(0), num_stacks_(3), recompute_backward_(true), remember_states_(false) {}
};




struct OptimOptions {
 OptimType type_;
 float eta_;
  float lambda_;
 float eta_decay_;
 float momentum_;
 float beta1_;
 float beta2_;
 float epsilon_;
 bool update_;
 OptimOptions():type_(SG), eta_(1), eta_decay_(1),
                momentum_(0.95), beta1_(0.9), beta2_(0.99),
                epsilon_(0.0001) {}
  OptimType type() { return type_; }
  float eta() { return eta_; }
  float lambda() { return lambda_; }
  bool update() { return update_; }
  float eta_decay() { return eta_decay_; }
  float momentum() { return momentum_; }
  float beta1() { return beta1_; }
  float beta2() { return beta2_; }
  float epsilon() { return epsilon_; }

  void set_update(bool update) { update_ = update; }
  void set_type(OptimType type) { type_ = type; }
  void set_eta(float eta) { eta_ = eta; }
  void set_lambda(float lambda) { lambda_ = lambda; }
  void set_eta_decay(float eta_decay) { eta_decay_ = eta_decay; }
  void set_momentum(float momentum) { momentum_ =  momentum; }
  void set_beta1(float beta1) { beta1_ = beta1; }
  void set_beta2(float beta2) { beta2_ = beta2; }
  void set_epsilon(float epsilon) { epsilon_ = epsilon; }
};



#endif
