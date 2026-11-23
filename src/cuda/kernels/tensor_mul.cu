#include "..\..\include\cuda\kernels.cuh"

constexpr uint32_t TensorSize_k = 32u;

__global__ void tensor_mul(const float* A, const float* B, float* C, int N)
{
	__shared__ half A_cache[TensorSize_k][TileSize_k];
	__shared__ half B_cache[TileSize_k][TensorSize_k];

	/*
								Calculate thread location in global memory
	*/

	// Thread index (0, 127) Linear threads in a vector
	auto thread_id = threadIdx.x;

	// Get mapped to 4 Warp (0 - 4)
	auto warp_id = thread_id / 32;

	// Make a 2x2 grid of warps
	auto warp_i = (warp_id / 2u);
	auto warp_j = (warp_id % 2u);

	const auto block_base_x = blockIdx.x * TensorSize_k;
	const auto block_base_y = blockIdx.y * TensorSize_k;

	wmma::fragment<wmma::accumulator, TileSize_k, TileSize_k, TileSize_k, float> acc;
	wmma::fill_fragment(acc, 0.0f);

	/*
							Populate Scratch Pad Memory (Shared Memory or LDS)
	*/

	// Tiles to scroll over the matrix
	auto num_tiles = (N + TileSize_k - 1u) / TileSize_k;

	for (auto t = 0u; t < num_tiles; ++t) {
		
		// the 4 warps
		for (auto w = 0u; w < 4u; ++w) {
			auto warp_index = thread_id + w * 128u;

			/*
					Matrix A -> 32x16
			*/

			// Map linear threads to a 2d block (tile)
			auto i = warp_index / TileSize_k;
			auto j = warp_index % TileSize_k;

			// map to the global matrix
			auto vram_i = block_base_y + i;
			auto vram_j = t * TileSize_k + j;

			// Slide tile across the X-axis
			if (vram_i < N && vram_j < N)
				A_cache[i][j] = __float2half(A[vram_i * N + vram_j]);
			else
				A_cache[i][j] = __float2half(0.0f);

			/*
					Matrix B -> 16x32
			*/

			// Map linear threads to a 2d block (tile)
			i = warp_index / TensorSize_k;
			j = warp_index % TensorSize_k;

			// map to the global matrix
			vram_i = t * TileSize_k + i;
			vram_j = block_base_x + j;

			// Slide tile across the Y-axis
			if (vram_i < N && vram_j < N)
				B_cache[i][j] = __float2half(B[vram_i * N + vram_j]);
			else
				B_cache[i][j] = __float2half(0.0f);
		}

		// Make sure all thread poluated the cache before moving forward
		__syncthreads();

		// Tensor Core Register (they are actually matrices)
		wmma::fragment<wmma::matrix_a, TileSize_k, TileSize_k, TileSize_k, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, TileSize_k, TileSize_k, TileSize_k, half, wmma::row_major> b_frag;

		// We need to address in granuality of the shared memory, so the stride is always the tile
		auto mem_offset = TileSize_k;

		// Load into Registers
		wmma::load_matrix_sync(a_frag, &A_cache[warp_i * mem_offset][0], TileSize_k);
		wmma::load_matrix_sync(b_frag, &B_cache[0][warp_j * mem_offset], TensorSize_k);


		// Multiply with ASIC hardware
		wmma::mma_sync(acc, a_frag, b_frag, acc);

		// Wait everyone before going for next tile
		__syncthreads();
	}

	// offset in vram
	auto warp_offset_i = warp_i * TileSize_k;
	auto warp_offset_j = warp_j * TileSize_k;

	auto warp_vram_i = block_base_y + warp_offset_i;
	auto warp_vram_j = block_base_x + warp_offset_j;

	// Check boundaries
	if (warp_vram_i < N && warp_vram_j < N)
	{
		float* C_address = &C[warp_vram_i * N + warp_vram_j];

		// write back a 16x16 block
		wmma::store_matrix_sync(C_address, acc, N, wmma::mem_row_major);
	}
}