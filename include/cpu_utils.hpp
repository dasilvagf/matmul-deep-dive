#pragma once

#include <random>
#include <algorithm>
#include <string>
#include <fstream>

inline void GenerateUniformDistribuitedElements(float* hostA, float* hostB, uint64_t size, float range_min, float range_max)
{
	/*
								Pseudo random number generator
	*/
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(range_min, range_max);

	/*
								Fill Matrix
	*/
	auto num_elements = size * size;
	std::generate(hostA, hostA + num_elements, [&]() { return dis(gen); });
	std::generate(hostB, hostB + num_elements, [&]() { return dis(gen); });
}

inline void DumpMatrixDataToDisk(const float* matrix, size_t matrix_bytes, const std::string& filename)
{
	std::ofstream file(filename, std::ios::binary);
	if (file)
	{
		file.write(reinterpret_cast<const char*>(matrix), matrix_bytes);
		file.close();
	}
}

inline void LoadMatrixDataFromDisk(float* matrix, size_t matrix_bytes, const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (file)
	{
		file.read(reinterpret_cast<char*>(matrix), matrix_bytes);
		file.close();
	}
	else
	{
		std::cerr << "CRITICAL ERROR: Input file " << filename << " could not be read!\n";
	}
}

inline bool ValidateMatrixUniform(const float* hostA, const float* hostB, const float* hostC, uint64_t size, uint32_t sample_size)
{
	/*
							Generate uniform indices 
	*/
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(0u, size - 1u);


	/*
							Sample the Matrix
	*/
	uint32_t erros = 0u;
	for (auto s = 0u; s < sample_size; ++s) {
		
		// Random point 
		int i = dis(gen);
		int j = dis(gen);

		// Dot product
		float reference = 0.0f;
		for (auto k = 0u; k < size; ++k) {
			reference += hostA[i * size + k] * hostB[k * size + j];
		}
	
		// Compare
		float actual = hostC[i * size + j];

		// tolerance of 0.1%
		float tolerance = std::max(std::abs(reference), std::abs(actual)) * 0.001f;

		// acceptable error e-3f
		float diff = std::abs(reference - actual);

		// relative error
		if (diff > tolerance)
			return false;
	}

	return true;
}

inline bool ValidateMatrixStratified(const float* hostA, const float* hostB, const float* hostC, uint64_t size, uint32_t strata_size)
{

	return true;
}
