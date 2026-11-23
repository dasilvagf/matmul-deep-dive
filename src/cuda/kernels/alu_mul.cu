#include "..\..\include\cuda\kernels.cuh"

__global__ void naive_mul(const float* A, const float* B, float* C, int N)
{
	/*
									Calculate thread location in global memory
	*/
	const auto block_base_x = blockIdx.x * blockDim.x;
	const auto block_base_y = blockIdx.y * blockDim.y;
	
	auto i = block_base_y + threadIdx.y;
	auto j = block_base_x + threadIdx.x;
	
	//auto i = blockIdx.x * blockDim.x + threadIdx.x;
	//	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	/*
									Actual math
	*/

	// Check boundaries (in case the matrix dimmension is not a multiple of 32)
	if (i < N && j < N)
	{
		float acc = 0.0f;

		// Dot product
		for (auto k = 0u; k < N; ++k) {
			// Read from global memory (Cache/VRAM) - SLOW

			// row 
			float a = A[i * N + k];
			float b = B[k * N + j];

			// accumulate
			acc += a * b;
		}

		// Write result to global memory (Cache/VRAM) - SLOW
		C[i * N + j] = acc;
	}
}

__global__ void tiled_mul(const float* A, const float* B, float* C, int N)
{
	__shared__ float A_cache[TileSize_k][TileSize_k];
	__shared__ float B_cache[TileSize_k][TileSize_k];

	/*
								Calculate thread location in global memory
	*/
	const auto block_base_x = blockIdx.x * blockDim.x;
	const auto block_base_y = blockIdx.y * blockDim.y;

	auto tx = threadIdx.x; // column	j
	auto ty = threadIdx.y; // row i

	auto i = block_base_y + ty;
	auto j = block_base_x + tx;

	float acc = 0.0f;

	/*	
							Populate Scratch Pad Memory (Shared Memory or LDS)
	*/

	// Tiles to scroll over the matrix
	auto num_tiles = (N + TileSize_k - 1u) / TileSize_k;
	for (auto t = 0u; t < num_tiles; ++t) {

		// Position in the tile in reference to global matrix
		auto tile_ty = t * TileSize_k + ty;
		auto tile_tx = t * TileSize_k + tx;

		// Slide tile across the X-axis
		if (i < N && tile_tx < N)
			A_cache[ty][tx] = A[i * N + tile_tx];
		else
			A_cache[ty][tx] = 0.0f;

		// Slide tile across the Y-axis
		if (j < N && tile_ty < N)
			B_cache[ty][tx] = B[tile_ty * N + j];
		else
			B_cache[ty][tx] = 0.0f;

		// Make sure all thread poluated the cache before moving forward
		__syncthreads();


#pragma unroll
		for (auto k = 0u; k < TileSize_k; ++k) {
			acc += A_cache[ty][k] * B_cache[k][tx];
		}

		// Wait everyone before going for next tile
		__syncthreads();
	}

	// Check boundaries (in case the matrix dimmension is not a multiple of 32)
	if (i < N && j < N)
	{
		C[i * N + j] = acc;
	}
}