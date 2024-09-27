#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>

#define TILE_WIDTH 32

void error_check(cudaError_t& err, int line_no, const char* file_name){
    if (err!=cudaSuccess) {
        std::cout<<cudaGetErrorString(err)<<" at line "<<line_no<<" in file "<<file_name<<std::endl;
    }
}


__global__ void matmul_kernel(float* M1, float* M2, float* M_Out,  unsigned int left_dim, unsigned int inner_dim, unsigned int right_dim){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row<left_dim && col<right_dim){
        float inner_product = 0.0f;
        for(int i=0; i<inner_dim; ++i){
            inner_product += M1[row*inner_dim+i]*M2[i*right_dim+col];
        }
        M_Out[row*right_dim+col] = inner_product;
    }
}


__global__ void matmul_kernel_row(float* M1, float* M2, float* M_Out,  unsigned int left_dim, unsigned int inner_dim, unsigned int right_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row <left_dim){
        for(int col=0; col<right_dim; ++col){
            float inner_product=0.0f;
            for(int i=0; i<inner_dim; ++i){
                inner_product += M1[row*inner_dim+i]*M2[i*right_dim+col];
            }
            M_Out[row*right_dim+col] = inner_product;
        }
    }
}


__global__ void matmul_kernel_col(float* M1, float* M2, float* M_Out,  unsigned int left_dim, unsigned int inner_dim, unsigned int right_dim){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col <right_dim){
        for(int row=0; row<left_dim; ++row){
            float inner_product=0.0f;
            for(int i=0; i<inner_dim; ++i){
                inner_product += M1[row*inner_dim+i]*M2[i*right_dim+col];
            }
            M_Out[row*right_dim+col] = inner_product;
        }
    }
}

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


void matmul(float* M1_h, float* M2_h, float* M_Out_h, unsigned int& left_dim, unsigned int& inner_dim, unsigned int& right_dim){
    float *M1_d, *M2_d, *M_Out_d;
    unsigned int m1_size = left_dim*inner_dim*sizeof(float), m2_size=inner_dim*right_dim*sizeof(float), m_out_size=left_dim*right_dim*sizeof(float);

    cudaError_t err ;

    // Allocating CUDA Memory
    err = cudaMalloc((void **)&M1_d, m1_size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&M2_d, m2_size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&M_Out_d, m_out_size);
    error_check(err, __LINE__, __FILE__);

    // Copying Data From Host To Device
    err = cudaMemcpy(M1_d, M1_h, m1_size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);
    err = cudaMemcpy(M2_d, M2_h, m2_size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);

    // Calling Kernel
    /*dim3 dimBlock(32, 32);
    dim3 dimGrid((int)std::ceil(right_dim/32.0), (int)std::ceil(left_dim/32.0));
    matmul_kernel<<<dimGrid, dimBlock>>>(M1_d, M2_d, M_Out_d, left_dim, inner_dim, right_dim);*/

    /*matmul_kernel_row<<<(int)std::ceil(left_dim/256.0f), 256>>>(M1_d, M2_d, M_Out_d, left_dim, inner_dim, right_dim); */

    /*matmul_kernel_col<<<(int)std::ceil(right_dim/256.0f), 256>>>(M1_d, M2_d, M_Out_d, left_dim, inner_dim, right_dim);*/

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((int)std::ceil(right_dim/TILE_WIDTH), (int)std::ceil(left_dim/TILE_WIDTH));
    matmul_kernel_tiled<<<dimGrid, dimBlock>>>(M1_d, M2_d, M_Out_d, left_dim, inner_dim, right_dim);

    err = cudaGetLastError();
    error_check(err, __LINE__, __FILE__);

    // Copying Output from Device to Host
    err = cudaMemcpy(M_Out_h, M_Out_d, m_out_size, cudaMemcpyDeviceToHost);
    error_check(err, __LINE__, __FILE__);

    // Freeing CUDA Memeory
    err = cudaFree(M1_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(M2_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(M_Out_d);
    error_check(err, __LINE__, __FILE__);
}


int main(){
    unsigned int left_dim, inner_dim, right_dim;

    std::cout<<"Enter the Left Dimension:";
    std::cin>>left_dim;

    std::cout<<"Enter the Inner Dimension:";
    std::cin>>inner_dim;

    std::cout<<"Enter the Right Dimension:";
    std::cin>>right_dim;

    std::vector<float> M1(left_dim*inner_dim), M2(inner_dim*right_dim), M_Out(left_dim*right_dim);

    // Generate Random Numbers
    std::mt19937 gen(666);
    std::normal_distribution<float> N(-1.0, 1.0);
    std::generate(std::begin(M1), std::end(M1), std::bind(N, gen));
    std::generate(std::begin(M2), std::end(M2), std::bind(N, gen));

    auto t0 = std::chrono::system_clock::now();

    matmul(M1.data(), M2.data(), M_Out.data(), left_dim, inner_dim, right_dim);

    auto t1 = std::chrono::system_clock::now();
    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout <<"Elapsed time is:"<<elapsed_par.count() << std::endl;

    return 0;
}