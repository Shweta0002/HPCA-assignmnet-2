#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
using namespace std;

#define THREADS_PER_BLOCK 16
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
__global__ void dilatedConvolutionKernel(int input_row,
    int input_col,
    int* input,
    int kernel_row,
    int kernel_col,
    int* kernel,
    int output_row,
    int output_col,
    long long unsigned int* output)
{



    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (output_i < output_row && output_j < output_col)
    {
        for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
        {
            for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
            {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_j = (output_j + 2 * kernel_j) % input_col;
                output[output_i * output_col + output_j] += input[input_i * input_col + input_j]
                    * kernel[kernel_i * kernel_col + kernel_j];
            }
        }
    }
}

void gpuThread(int input_row,
    int input_col,
    int* input,
    int kernel_row,
    int kernel_col,
    int* kernel,
    int output_row,
    int output_col,
    long long unsigned int* output)
{
    // Allocate memory on the device
    int* deviceInput, * deviceKernel;
    long long unsigned int* deviceOutput;

    gpuErrchk(cudaMalloc((void**)&deviceInput, sizeof(int) * input_row * input_col));
    gpuErrchk(cudaMalloc((void**)&deviceKernel, sizeof(int) * kernel_row * kernel_col));
    gpuErrchk(cudaMalloc((void**)&deviceOutput, sizeof(long long unsigned int) * output_row * output_col));

    // Copy input and kernel matrices from host to device
    gpuErrchk(cudaMemcpy(deviceInput, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(deviceKernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((output_col + blockSize.x - 1) / blockSize.x, (output_row + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    dilatedConvolutionKernel << <gridSize, blockSize >> > (input_row, input_col, deviceInput, kernel_row, kernel_col, deviceKernel, output_row, output_col, deviceOutput);

    // Copy the result from device to host
    gpuErrchk(cudaMemcpy(output, deviceOutput, sizeof(long long unsigned int) * output_row * output_col, cudaMemcpyDeviceToHost));

    // Free allocated memory on the device
    gpuErrchk(cudaFree(deviceInput));
    gpuErrchk(cudaFree(deviceKernel));
    gpuErrchk(cudaFree(deviceOutput));

    
}
