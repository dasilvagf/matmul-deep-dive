#include <array>
#include <iostream>

#include "..\include\cpu_utils.hpp"
#include "..\include\cuda\kernels.cuh"

// Define Test Cases
enum TestCasesEnum
{
    SmokeTest = 0u,
    BenchmarkTest,
    StressTest,
    MassiveTest
};

// Size of the matrices for this test
constexpr uint32_t MatrixSizeSmokeTest_k   = 2048u;
constexpr uint32_t MatrixSizeBenchmark_k   = 8192u;
constexpr uint32_t MatrixSizeStressTest_k  = 16384u;
constexpr uint32_t MatrixSizeMassiveTest_k = 24576u;

// Matrix size in linear addresses
constexpr size_t MatrixBytesSmoke_k   = MatrixSizeSmokeTest_k   * MatrixSizeSmokeTest_k   * sizeof(float); // 16   Mib
constexpr size_t MatrixBytesBench_k   = MatrixSizeBenchmark_k   * MatrixSizeBenchmark_k   * sizeof(float); // 256  Mib
constexpr size_t MatrixBytesStress_k  = MatrixSizeStressTest_k  * MatrixSizeStressTest_k  * sizeof(float); // 1    Gib
constexpr size_t MatrixBytesMassive_k = MatrixSizeMassiveTest_k * MatrixSizeMassiveTest_k * sizeof(float); // 2.25 Gib

constexpr std::array<uint32_t,    4> MatrixSizesArray = { MatrixSizeSmokeTest_k, MatrixSizeBenchmark_k, MatrixSizeStressTest_k, MatrixSizeMassiveTest_k };
constexpr std::array<size_t,      4> MatrixBytesArray = { MatrixBytesSmoke_k,    MatrixBytesBench_k,    MatrixBytesStress_k,    MatrixBytesMassive_k };

int main(int argc, char** argv)
{
    // Select which type of test we running
    uint8_t testType = TestCasesEnum::BenchmarkTest;
    if (argc == 2u)
        testType = static_cast<uint8_t>(std::stoi(argv[1]));

    // Select Matrix Size
    uint32_t matrixSize  = MatrixSizesArray[testType];
    size_t   matrixBytes = MatrixBytesArray[testType];

    /*
                                        Host
    */
    float* hostA = nullptr;
    float* hostB = nullptr;
    float* hostC = nullptr;

    // 
    //  Allocation of the data in the Host RAM is made in a Pinned Page. 
    //  This way the OS dont swap our pages into Disk/RAM.
    //
    cudaMallocHost(&hostA, matrixBytes);
    cudaMallocHost(&hostB, matrixBytes);
    cudaMallocHost(&hostC, matrixBytes);

    //
    //  Input data
    //
    if (testType == TestCasesEnum::StressTest)
    {
        LoadMatrixDataFromDisk(hostA, matrixSize, "data\\hostA-stress.bin");
        LoadMatrixDataFromDisk(hostB, matrixSize, "data\\hostB-stress.bin");
    }
    else if (testType == TestCasesEnum::MassiveTest)
    {
        LoadMatrixDataFromDisk(hostA, matrixSize, "data\\hostA-massive.bin");
        LoadMatrixDataFromDisk(hostB, matrixSize, "data\\hostB-massive.bin");
    }
    else
    {
        // For smaller data sets we generate the numbers on the fly
        GenerateUniformDistribuitedElements(hostA, hostB, matrixSize, 0.0f, 1.0f);
    }

    /*
                                        Device
    */
    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    
    // Allocation into the GPU VRAM (GDDR/HBM)
    cudaMalloc(&deviceA, matrixBytes);
    cudaMalloc(&deviceB, matrixBytes);
    cudaMalloc(&deviceC, matrixBytes);

    /*
                                        PCI Bus In
    */

    // Copy input matrices
    cudaMemcpy(deviceA, hostA, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, matrixBytes, cudaMemcpyHostToDevice);

    /*
                                        GPU Computation
    */

    // Algorithmic-View (Tiling)
   // const auto matrixTileSize = 16u;
     const auto matrixTileSize = 32u;
    const auto matrixNumberOfTilesCelling = (matrixSize + matrixTileSize - 1u) / matrixTileSize;

    // Silicon-View
    dim3 numBlocks(matrixNumberOfTilesCelling, matrixNumberOfTilesCelling);
    //dim3 threadsPerBlock(16u, 16u);
    dim3 threadsPerBlock(128u, 1u);

    // 
    // Profilling using GPU Counters
    //
    cudaEvent_t startTick;
    cudaEvent_t stopTick;

    cudaEventCreate(&startTick);
    cudaEventCreate(&stopTick);

    //
    // Execute GPU Kernel
    //
    cudaEventRecord(startTick);
   
    //tiled_mul<<<numBlocks, threadsPerBlock>>> (deviceA, deviceB, deviceC, matrixSize);
    //tensor_mul<<<numBlocks, threadsPerBlock >>> (deviceA, deviceB, deviceC, matrixSize);
    strassen_alu_mul(deviceA, deviceB, deviceC, matrixSize);

    // No need to sync, this call gets added to the GPU Command Circular Queue right after Kernel execution.
    cudaEventRecord(stopTick);

    /*
                                        Validation
    */

    // Barrier for CPU-GPU sync
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err);
    }

    // Get result back from the GPU
    cudaMemcpy(hostC, deviceC, matrixBytes, cudaMemcpyDeviceToHost);

    // Use Stratified Sampling to verify the result (Jittering)
    if (ValidateMatrixUniform(hostA, hostB, hostC, matrixSize, matrixSize))
    {
        std::cout << "VALIDATION: Success! No errors were produced during the computation." << std::endl;
    }
    else
    {
        std::cerr << "VALIDATION: Failuire! The resulted computation do not match the reference." << std::endl;
    }

    /*
                                        Profilling
    */

    // Measure the time elapsed by the use the GPU clock cycles
    float msElapsed = 0.0f;
    cudaEventElapsedTime(&msElapsed, startTick, stopTick);

    std::cout << "==================================================================" << std::endl;
    std::cout << "============================= PROFILING ==========================" << std::endl;
    std::cout << "GPU Elapsed Time (Tiled Mul): " << msElapsed << " (ms)"             << std::endl;
    std::cout << "==================================================================" << std::endl;


    /*  
                                        Clean
    */
    if (deviceA)
        cudaFree(deviceA);
    if (deviceB)
        cudaFree(deviceB);
    if (deviceC)
        cudaFree(deviceC);

    if (hostA)
        cudaFreeHost(hostA);
    if (hostB)
        cudaFreeHost(hostB);
    if (hostC)
        cudaFreeHost(hostC);

    return 0x0;
}