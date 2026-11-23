#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

constexpr uint32_t TileSize_k = 16u;

// Multiplication using the ALUs and completely ignoring GPU Architecture
__global__ void naive_mul(const float* A, const float* B, float* C, int N);

// Multiplication using the ALUs and memory hierarchy
__global__ void tiled_mul(const float* A, const float* B, float* C, int N);

// Multiplication using the Tensor Cores and memory hierarchy
__global__ void tensor_mul(const float* A, const float* B, float* C, int N);

// Multiplication using Strassen and ALUs
void strassen_alu_mul(const float* A, const float* B, float* C, int N);

// Multiplication using Strassen and Tensor Cores
void strassen_tensor_mul(const float* A, const float* B, float* C, int N);