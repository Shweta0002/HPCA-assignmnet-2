void singleThread( int input_row, 
                int input_col,
                int *input, 
                int kernel_row, 
                int kernel_col, 
                int *kernel,
                int output_row, 
                int output_col, 
                long long unsigned int *output ) 
{
    for(int output_i = 0; output_i< output_row; output_i++)
    {
        int output_offset = output_i * output_col;

        for(int output_j = 0; output_j< output_col; output_j += 4)
        {
            long long unsigned int res0 = 0;
            long long unsigned int res1 = 0;
            long long unsigned int res2 = 0;
            long long unsigned int res3 = 0;

            int input_i = output_i - 2;

            for(int kernel_i = 0; kernel_i< kernel_row; kernel_i++)
            {
                int kernel_offset = kernel_i * kernel_col;

                if(input_i + 2 >= input_row) 
                { 
                    if(input_i + 2 == input_col) 
                        input_i = 0;
                    else 
                        input_i = 1;
                }
                else 
                    input_i = (input_i + 2);


                int input_offset = input_i*input_col;

                int input_j = output_j - 2;

                for(int kernel_j = 0; kernel_j< kernel_col; kernel_j++)
                {
                    if(input_j +2 >= input_col) 
                    { 
                        if(input_j+2==input_col) 
                            input_j=0;
                        else 
                            input_j=1;
                    }

                    else 
                        input_j=(input_j+2);
  
                    res0 += input[ input_offset + input_j] * kernel[kernel_offset + kernel_j];
                    res1 += input[ input_offset + (input_j+1)%input_col] * kernel[kernel_offset + kernel_j];
                    res2 += input[ input_offset + (input_j+2)%input_col] * kernel[kernel_offset + kernel_j];
                    res3 += input[ input_offset + (input_j+3)%input_col] * kernel[kernel_offset + kernel_j];
                
                }
            }

            output[output_offset + output_j] = res0;
            if(output_j + 1 >= output_col) continue;
            output[output_offset + output_j + 1] = res1;
            if(output_j + 2 >= output_col) continue;
            output[output_offset + output_j + 2] = res2;
            if(output_j + 3 >= output_col) continue;
            output[output_offset + output_j + 3] = res3;

        }
    }
}
