__global__ void mat_vec_mul_kernel_tiled(float* M, float* V, float* Out, unsigned int num_rows, unsigned int num_cols){

    // Creating Buffers in shared memory to store data.
    __shared__ float Vs[TILE_WIDTH];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float inner_product=0.0f;
    for(int phase=0; phase<(num_cols+TILE_WIDTH-1)/TILE_WIDTH; ++phase){
        if((phase*TILE_WIDTH+threadIdx.x)<num_cols){
            Vs[threadIdx.x] = V[phase*TILE_WIDTH+threadIdx.x];
        }
        else{
            Vs[threadIdx.x] = 0.0f;
        }
        __syncthreads(); //Synchronization

        for(int i=0; i<TILE_WIDTH; ++i){
            // To Prevent Out-of-bound memory access for M
            if(phase*TILE_WIDTH+i<num_cols){
                inner_product += M[idx*num_cols+phase*TILE_WIDTH+i]*Vs[i];
            }
        }
        __syncthreads(); //Synchronization
    }

    if(idx<num_rows){
        Out[idx] = inner_product;
    }
}