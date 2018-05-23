#include "arena.h"
#include "syncedmem.h"
#include "test_main.h"
#include "math_functions.h"

class SyncedMemoryTest: public::testing::Test {};
TEST_F(SyncedMemoryTest, TestInitialization) {
  int n = 10;
  SyncedMemory mem(n);
  EXPECT_EQ(mem.head(), SyncedMemory::UNINITIALIZED);
  EXPECT_EQ(mem.size(), n);
  SyncedMemory *p_mem = new SyncedMemory(n);
  EXPECT_EQ(p_mem->size(), n);
  delete  p_mem;
}

TEST_F(SyncedMemoryTest, TestAllocation) {
  int n = 10;
  SyncedMemory mem(n);
  EXPECT_TRUE(mem.cpu_data());
  EXPECT_TRUE(mem.gpu_data());
  EXPECT_TRUE(mem.mutable_cpu_data());
  EXPECT_TRUE(mem.mutable_gpu_data());
}

TEST_F(SyncedMemoryTest, TestCPUWrite) {
  int n = 10;
  SyncedMemory mem(n);
  float* cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  cpu_fill(mem.size(), 1, cpu_data);
  for(int i = 0; i<mem.size(); i++) {
    EXPECT_EQ(cpu_data[i], 1);
  }
  const float* gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  float *recovered_value = new float [mem.size()];
  cudaMemcpy(recovered_value, gpu_data, mem.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0; i<mem.size(); i++) {
    EXPECT_EQ(recovered_value[i], float(1));
  }
  cpu_data = mem.mutable_cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_CPU);
  cpu_fill(mem.size(), 2, cpu_data);
  for(int i=0; i<mem.size(); i++) {
    EXPECT_EQ(cpu_data[i], 2);
  }
  gpu_data = mem.gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  cudaMemcpy(recovered_value, gpu_data, mem.size() * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i=0; i<mem.size(); i++) {
    EXPECT_EQ(recovered_value[i], 2);
  }
  delete[] recovered_value;
}


TEST_F(SyncedMemoryTest, TestGPUWrite) {
  int n=10;
  SyncedMemory mem(n);
  float *gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  gpu_fill(mem.size(), 1, gpu_data);

  const float *cpu_data = mem.cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  for(int i=0; i<mem.size(); i++) {
    EXPECT_EQ(cpu_data[i], 1);
  }
  gpu_data = mem.mutable_gpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::HEAD_AT_GPU);
  gpu_fill(mem.size(), 2, gpu_data);

  cpu_data = mem.cpu_data();
  EXPECT_EQ(mem.head(), SyncedMemory::SYNCED);
  for(int i=0; i<mem.size(); i++) {
    EXPECT_EQ(cpu_data[i], 2);
  }
}
