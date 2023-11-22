#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <string.h>
using namespace std;

#define THREADS_PER_BLOCK 16
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void dilatedConvolutionKernel(int input_row,
                                         int input_col,
                                         int *input,
                                         int kernel_row,
                                         int kernel_col,
                                         int *kernel,
                                         int output_row,
                                         int output_col,
                                         long long unsigned int *output)
{
    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;
  


    if (output_i < output_row && output_j < output_col)
    {
         output[output_i * output_col + output_j] = 0;
       for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    int input_i = (output_i + 2*kernel_i) % input_row;
                    int input_j = (output_j + 2*kernel_j) % input_col;
                    output[output_i * output_col + output_j] += input[input_i*input_col +input_j] 
                                                                * kernel[kernel_i*kernel_col +kernel_j];
                }
            }
    }
}

void gpuThread(int input_row,
               int input_col,
               int *input,
               int kernel_row,
               int kernel_col,
               int *kernel,
               int output_row,
               int output_col,
               long long unsigned int *output)
{
    int *d_input, *d_kernel;
    long long unsigned int *d_output;
    gpuErrchk(cudaMalloc((void **)&d_input, sizeof(int) * input_row * input_col));
    gpuErrchk(cudaMalloc((void **)&d_kernel, sizeof(int) * kernel_row * kernel_col));
    gpuErrchk(cudaMalloc((void **)&d_output, sizeof(long long unsigned int) * output_row * output_col));

    std::cout << "hello" << std::endl;
    // Copy input and kernel to device
    gpuErrchk(cudaMemcpy(d_input, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockDim(2, 2);
    dim3 gridDim((output_row + blockDim.x - 1) / blockDim.x, (output_col + blockDim.y - 1) / blockDim.y);

    dilatedConvolutionKernel<<<gridDim, blockDim>>>(input_row, input_col, d_input, kernel_row, kernel_col, d_kernel, output_row, output_col, d_output);

    // Copy result back to host
    gpuErrchk(cudaMemcpy(output, d_output, sizeof(int) * output_row * output_col, cudaMemcpyDeviceToHost));

    // Free device memory
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_kernel));
    gpuErrchk(cudaFree(d_output));
}
