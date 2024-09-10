#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>


void error_check(cudaError_t& err, int line_no, const char* file_name){
    if (err!=cudaSuccess) {
        std::cout<<cudaGetErrorString(err)<<" at line "<<line_no<<" in file "<<file_name<<std::endl;
    }
}


__global__ void vec_add_kernel(float* A, float* B, float* C, unsigned int n){
    unsigned int i = (blockIdx.x * blockDim.x)+threadIdx.x;
    if (i<n){
        C[i] = A[i]+B[i];
    }
}

void vec_add(float* A_h, float* B_h, float* C_h, unsigned int& n){
    float *A_d, *B_d, *C_d;
    unsigned int size = n * sizeof(float);

    cudaError_t err ;

    // Allocating CUDA Memory
    err = cudaMalloc((void **)&A_d, size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&B_d, size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&C_d, size);
    error_check(err, __LINE__, __FILE__);


    // Copying Data From Host To Device
    err = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);
    err = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);

    // Calling Kernel
    vec_add_kernel<<<(int)std::ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    err = cudaGetLastError();
    error_check(err, __LINE__, __FILE__);

    // Copying Output from Device to Host
    err = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    error_check(err, __LINE__, __FILE__);

    // Freeing CUDA Memeory
    err = cudaFree(A_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(B_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(C_d);
    error_check(err, __LINE__, __FILE__);

}

int main(){

    unsigned int vec_size;

    std::cout<<"Enter the size of the vector:";
    std::cin>>vec_size;

    std::vector<float> A(vec_size), B(vec_size), C(vec_size);

    // Generate Random Numbers
    std::mt19937 gen(666);
    std::normal_distribution<float> N(-1.0, 1.0);
    std::generate(std::begin(A), std::end(A), std::bind(N, gen));
    std::generate(std::begin(B), std::end(B), std::bind(N, gen));

    auto t0 = std::chrono::system_clock::now();
    vec_add(A.data(), B.data(), C.data(), vec_size);

    auto t1 = std::chrono::system_clock::now();

    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout <<"Elapsed time is:"<<elapsed_par.count() << std::endl;
    return 0;
}