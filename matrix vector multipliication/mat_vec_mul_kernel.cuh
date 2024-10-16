__global__ void mat_vec_mul_kernel(float* M, float* V, float* Out, unsigned int num_rows, unsigned int num_cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<num_rows){
        Out[idx] = 0;
        for(int i=0; i<num_cols; ++i){
            Out[idx] += M[idx*num_cols+i]*V[i];
        }
    }
}