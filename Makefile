PROJECT := krf
TEST_GPUID := 0
include Makefile.config
NAME := lib$(PROJECT).so
STATIC_NAME := lib$(PROJECT).a

SRC_DIR := src
# CXX_SRCS are the source files excluding the test ones.
CXX_SRCS := $(shell find $(SRC_DIR) ! -name "test_*.cc" -name "*.cc")
# HXX_SRCS are the header files
CXX_HDRS := $(shell find $(SRC_DIR) ! -name "*.h")
# CU_SRCS are the cuda source files
CU_SRCS := $(shell find $(SRC_DIR)  -name "*.cu")
# TEST_SRCS are the test source files
TEST_SRCS := $(shell find $(SRC_DIR)/test  -name "test_*.cc")
GTEST_SRC := $(SRC_DIR)/gtest/gtest-all.cpp
EXAMPLE_SRCS := $(shell find examples  -name "*.cc")

# TEST_HDRS are the test header files
TEST_HDRS := $(shell find $(SRC_DIR)/test  -name "test_*.h")
# PROTO_SRCS are the protocol buffer definitions
PROTO_SRCS := $(wildcard $(SRC_DIR)/*.proto)
##############################
# Derive generated files
##############################
# The generated files for protocol buffers
#PROTO_GEN_HDRS := ${PROTO_SRCS:.proto=.pb.h}
#PROTO_GEN_CC := ${PROTO_SRCS:.proto=.pb.cc}
# The objects corresponding to the source files
# These objects will be linked into the final shared library, so we
# exclude the test and example objects.
CXX_OBJS  := $(CXX_SRCS:$(SRC_DIR)/%.cc=$(BUILD_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.cuo)
OBJS := $(CXX_OBJS) $(CU_OBJS)
TEST_OBJS  := $(TEST_SRCS:$(SRC_DIR)/test/%.cc=$(BUILD_DIR)/%.o)
GTEST_OBJ  := $(GTEST_SRC:$(SRC_DIR)/gtest/%.cpp=$(BUILD_DIR)/%.o)

EXAMPLE_OBJS  := $(EXAMPLE_SRCS:examples/%.cc=$(BUILD_DIR)/%.o)

# program and test bins
TEST_BINS := ${TEST_OBJS:.o=.testbin}
EXAMPLE_BINS := ${EXAMPLE_OBJS:.o=.bin}
##############################
# Derive include and lib directories
##############################
CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib $(CUDA_DIR)/lib64
BLAS_INCLUDE_DIR := $(BLAS_DIR)/include
BLAS_LIB_DIR := $(BLAS_DIR)/lib $(BLAS_DIR)/lib/intel64 

INCLUDE_DIRS += ./src  $(CUDA_INCLUDE_DIR) $(BLAS_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR) $(BLAS_LIB_DIR)
LIBRARIES :=  cudart cublas curand cusolver glog gflags openblas pthread
WARNINGS := -Wall

COMMON_FLAGS := -DNDEBUG -O2  $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread  -fPIC  $(COMMON_FLAGS) -std=c++11
NVCCFLAGS := -ccbin=$(CXX) -Xcompiler -fPIC  $(COMMON_FLAGS) -std=c++11
LDFLAGS +=  $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
	    $(foreach library,$(LIBRARIES),-l$(library))
##############################
# Define build targets
##############################
.PHONY: all init test clean linecount distribute

all: init $(NAME) $(STATIC_NAME)
	     @echo $(CXX_OBJS)

init:
	@ mkdir -p $(foreach obj,$(OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(TEST_OBJS),$(dir $(obj)))
	@ mkdir -p $(foreach obj,$(GTEST_OBJ),$(dir $(obj)))

linecount: clean
	cloc --read-lang-def=$(PROJECT).cloc src/

test: init $(TEST_BINS)

examples: init $(EXAMPLE_BINS)

$(NAME): init  $(OBJS)
	$(CXX) -fPIC -shared -o $(NAME) $(OBJS) $(LDFLAGS)
	@echo

$(STATIC_NAME): init  $(OBJS)
	ar rcs $(STATIC_NAME) $(PROTO_OBJS) $(OBJS)
	@echo

runtest: test
	for testbin in $(TEST_BINS); do $$testbin $(TEST_GPUID); done

$(TEST_BINS): %.testbin : %.o $(GTEST_OBJ) $(STATIC_NAME) $(TEST_HDRS)
	$(CXX) $< $(GTEST_OBJ) $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS) -fopenmp

$(EXAMPLE_BINS): %.bin : %.o  $(STATIC_NAME)
	$(CXX) $<  $(STATIC_NAME) -o $@ $(LDFLAGS) $(WARNINGS) -fopenmp

$(OBJS):  $(CXX_HDRS)

$(BUILD_DIR)/%.o: src/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

$(BUILD_DIR)/%.o: src/test/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

$(BUILD_DIR)/%.o: src/gtest/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

$(BUILD_DIR)/%.cuo: src/%.cu
	$(CUDA_DIR)/bin/nvcc $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@
	@echo

$(BUILD_DIR)/%.o: examples/%.cc
	$(CXX) $< $(CXXFLAGS) -c -o $@ $(LDFLAGS)
	@echo

#$(PROTO_GEN_CC): $(PROTO_SRCS)
#	protoc --proto_path=src --cpp_out=src $(PROTO_SRCS)
#	@echo

clean:
	@- $(RM) $(NAME) $(STATIC_NAME)
	@- $(RM) -rf $(BUILD_DIR)
	@echo "executable removed!"
