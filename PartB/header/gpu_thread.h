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
    int input_i, input_j;
    int value = 0;

    if (output_i < output_row && output_j < output_col)
    {
        int output_offset = output_i * output_col;
        input_i = output_i - 2;
        for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++)
        {
            int kernel_offset = kernel_i * kernel_col;

            if (input_i + 2 >= input_row)
            {
                if (input_i + 2 == input_col)
                    input_i = 0;
                else
                    input_i = 1;
            }
            else
                input_i = (input_i + 2);

            int input_offset = input_i * input_col;

            int input_j = output_j - 2;

            for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++)
            {
                if (input_j + 2 >= input_col)
                {
                    if (input_j + 2 == input_col)
                        input_j = 0;
                    else
                        input_j = 1;
                }

                else
                    input_j = (input_j + 2);

                value += input[input_offset + input_j] * kernel[kernel_offset + kernel_j];
            }
        }

        output[output_offset + output_j] = value;
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
    cudaMalloc((void **)&d_input, sizeof(int) * input_row * input_col);
    cudaMalloc((void **)&d_kernel, sizeof(int) * kernel_row * kernel_col);
    cudaMalloc((void **)&d_output, sizeof(int) * output_row * output_col);

    // Copy input and kernel to device
    cudaMemcpy(d_input, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(2, 2);
    dim3 gridDim((output_row + blockDim.x - 1) / blockDim.x, (output_col + blockDim.y - 1) / blockDim.y);

    dilatedConvolutionKernel<<<gridDim, blockDim>>>(input_row, input_col, d_input, kernel_row, kernel_col, d_kernel, output_row, output_col, d_output);

    // Copy result back to host
    cudaMemcpy(output, d_output, sizeof(int) * output_row * output_col, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}