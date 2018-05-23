#include "arena.h"

shared_ptr<Arena> Arena::singleton_;

// random seeding: This functions are from caffe
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }
  LOG(INFO) << "System entropy source not available, "
      "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);
  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

Arena::Arena()
    :device_(Arena::CPU), phase_(Arena::TRAIN), cublas_handle_(NULL),
     curand_generator_(NULL) {
  //random_generator_.reset(new Prand(cluster_seedgen()));
  random_generator_.reset(new Prand(1));
  if(cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle, Cublas won't be available";
  }
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
            != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, 1701ULL)
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
}

Arena::~Arena() {
  if (!cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (!curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
}
