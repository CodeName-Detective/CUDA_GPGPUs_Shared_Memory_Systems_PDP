#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>

#include "constants.hpp"
#include "mat_vec_mul_kernel.cuh"
#include "mat_vec_mul_kernel_tiled.cuh"


void error_check(cudaError_t& err, int line_no, const char* file_name){
    if (err!=cudaSuccess) {
        std::cout<<cudaGetErrorString(err)<<" at line "<<line_no<<" in file "<<file_name<<std::endl;
    }
}


void mat_vec_mul(float* M_h, float* V_h, float* Out_h, unsigned int& num_rows, unsigned int& num_cols){
    float *M_d, *V_d, *Out_d;

    unsigned int m_size = num_rows*num_cols*sizeof(float), v_size=num_cols*sizeof(float), out_size=num_rows*sizeof(float);

    cudaError_t err ;

    // Allocating CUDA Memory
    err = cudaMalloc((void **)&M_d, m_size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&V_d, v_size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&Out_d, out_size);
    error_check(err, __LINE__, __FILE__);

    // Copying Data From Host To Device
    err = cudaMemcpy(M_d, M_h, m_size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);
    err = cudaMemcpy(V_d, V_h, v_size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);

    // Calling Kernel
    /*mat_vec_mul_kernel<<<(int)std::ceil(num_rows/256.0f), 256>>>(M_d, V_d, Out_d, num_rows, num_cols);*/
    mat_vec_mul_kernel_tiled<<<(num_rows+TILE_WIDTH-1)/TILE_WIDTH), TILE_WIDTH>>>(M_d, V_d, Out_d, num_rows, num_cols);
    err = cudaGetLastError();
    error_check(err, __LINE__, __FILE__);

    // Copying Output from Device to Host
    err = cudaMemcpy(Out_h, Out_d, out_size, cudaMemcpyDeviceToHost);
    error_check(err, __LINE__, __FILE__);

    // Freeing CUDA Memeory
    err = cudaFree(M_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(V_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(Out_d);
    error_check(err, __LINE__, __FILE__);
}


int main(){
    unsigned int num_rows, num_cols;

    std::cout<<"Enter the Number of rows:";
    std::cin>>num_rows;

    std::cout<<"Enter the Number of columns:";
    std::cin>>num_cols;

    std::vector<float> M(num_rows*num_cols), V(num_cols), Out(num_rows);

    // Generate Random Numbers
    std::mt19937 gen(666);
    std::normal_distribution<float> N(-1.0, 1.0);
    std::generate(std::begin(M), std::end(M), std::bind(N, gen));
    std::generate(std::begin(V), std::end(V), std::bind(N, gen));

    auto t0 = std::chrono::system_clock::now();

    mat_vec_mul(M.data(), V.data(), Out.data(), num_rows, num_cols);

    auto t1 = std::chrono::system_clock::now();
    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout <<"Elapsed time is:"<<elapsed_par.count() << std::endl;

    return 0;
}