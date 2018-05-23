#include "arena.h"
#include "test_main.h"

class ArenaTest: public::testing::Test {};

TEST_F(ArenaTest, TestCublasHandler) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Arena::cublas_handle());
}

TEST_F(ArenaTest, TestBrewMode) {
  EXPECT_EQ(Arena::mode(), Arena::CPU);
  Arena::set_mode(Arena::GPU);
  EXPECT_EQ(Arena::mode(), Arena::GPU);
}

TEST_F(ArenaTest, TestPhaseMode) {
  EXPECT_EQ(Arena::phase(), Arena::TRAIN);
  Arena::set_phase(Arena::TEST);
  EXPECT_EQ(Arena::phase(), Arena::TEST);
}
