#include "constants.hpp"

__global__ void matmul_kernel_tiled(float* M1, float* M2, float* M_Out,  unsigned int left_dim, unsigned int inner_dim, unsigned int right_dim){

    // Creating Buffers in shared memory to store data.
    __shared__ float M1s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float M2s[TILE_WIDTH][TILE_WIDTH];

    // We can also use TILE_WIDTH instead of blockDim as both are same.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Loop Over THe Phases
    float inner_product=0.0f;
    for(int phase=0; phase<(inner_dim+TILE_WIDTH-1)/TILE_WIDTH; ++phase){

        // Loading Data into shared memory.
        //Load M1s - Handling Boundary Conditions
        if(row<left_dim && (phase*TILE_WIDTH+threadIdx.x)<inner_dim){
            M1s[threadIdx.y][threadIdx.x] = M1[(row*inner_dim)+(phase*TILE_WIDTH+threadIdx.x)];
        }
        else{
            M1s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        //Load M2s - Handling Boundary Conditions
        if((phase*TILE_WIDTH+threadIdx.y)<inner_dim && col<right_dim){
            M2s[threadIdx.y][threadIdx.x] = M2[((phase*TILE_WIDTH+threadIdx.y)*right_dim)+(col)];
        }
        else{
            M2s[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads(); //Synchronization

        // Performing Inner Product in 1 Phase for the Tile Loaded in Shared memory
        for(int i=0; i<TILE_WIDTH; ++i){
            inner_product += M1s[threadIdx.y][i]*M2s[i][threadIdx.x];
        }
        __syncthreads(); //Synchronization
    }
    //  Handling Boundary Conditions
    if(row<left_dim && col<right_dim){
        M_Out[row*right_dim+col] = inner_product;
    }
}