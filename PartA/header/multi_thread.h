#include <pthread.h>

#define MAX_THREADS 12
// Create other necessary functions here
typedef struct
{
    int input_row;
    int input_col;
    int *input;
    int kernel_row;
    int kernel_col;
    int *kernel;
    int output_row;
    int output_col;
    int start_row;
    int end_row;
    long long unsigned int *output;
} inputParameters;

void *thread_function(void *data)
{
    inputParameters *param = (inputParameters *)data;
    // for(int i = 0; i < param->output_col * param->output_col; ++i){
    //     param->output[i] = 0;
    // }
    int input_row = param->input_row;
    int input_col = param->input_col;
    int *input = param->input;
    int kernel_row = param->kernel_row;
    int kernel_col = param->kernel_col;
    int *kernel = param->kernel;
    int output_row = param->output_row;
    int output_col = param->output_col;
    long long unsigned int *output = param->output;

    for (int output_i = param->start_row; output_i < param->end_row; output_i++)
    {
        int output_offset = output_i * output_col;

        for (int output_j = 0; output_j < output_col; output_j += 4)
        {
            long long unsigned int res0 = 0;
            long long unsigned int res1 = 0;
            long long unsigned int res2 = 0;
            long long unsigned int res3 = 0;

            int input_i = output_i - 2;

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

                    res0 += input[input_offset + input_j] * kernel[kernel_offset + kernel_j];
                    res1 += input[input_offset + (input_j + 1) % input_col] * kernel[kernel_offset + kernel_j];
                    res2 += input[input_offset + (input_j + 2) % input_col] * kernel[kernel_offset + kernel_j];
                    res3 += input[input_offset + (input_j + 3) % input_col] * kernel[kernel_offset + kernel_j];
                }
            }

            output[output_offset + output_j] = res0;
            if (output_j + 1 >= output_col)
                continue;
            output[output_offset + output_j + 1] = res1;
            if (output_j + 2 >= output_col)
                continue;
            output[output_offset + output_j + 2] = res2;
            if (output_j + 3 >= output_col)
                continue;
            output[output_offset + output_j + 3] = res3;
        }
    }
    pthread_exit(NULL);
}

// Fill in this function
void multiThread(int input_row,
                 int input_col,
                 int *input,
                 int kernel_row,
                 int kernel_col,
                 int *kernel,
                 int output_row,
                 int output_col,
                 long long unsigned int *output)

{

    pthread_t threads[MAX_THREADS];
    inputParameters threadData[MAX_THREADS];
    int rows_per_thread = input_row / MAX_THREADS;
    int remaining_rows = input_row % MAX_THREADS;

    for (int i = 0; i < MAX_THREADS; i++)
    {
        threadData[i].input_row = input_row;
        threadData[i].input_col = input_col;
        threadData[i].input = input;
        threadData[i].kernel_row = kernel_row;
        threadData[i].kernel_col = kernel_col;
        threadData[i].kernel = kernel;
        threadData[i].output_row = output_row;
        threadData[i].output_col = output_col;
        threadData[i].output = output;
        threadData[i].start_row = i * rows_per_thread;
        threadData[i].end_row = (i + 1) * rows_per_thread;

        if (i == MAX_THREADS - 1)
        {
            // Assign the remaining rows to the last thread
            threadData[i].end_row += remaining_rows;
        }

        pthread_create(&threads[i], NULL, thread_function, (void *)&threadData[i]);
    }

    for (int i = 0; i < MAX_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
}
