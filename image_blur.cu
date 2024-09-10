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


__global__ void image_blur_kernel(float* input_image, float* output_image, unsigned int width, unsigned int height, unsigned int blur_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row<height && col<width){
        float neighbour_sum = 0.0f;
        int  num_neighbours = 0;
        for(int blur_row = - blur_size; blur_row<blur_size+1; ++blur_row){
            for(int blur_col = - blur_size; blur_col<blur_size+1; ++blur_col){
                int neighbour_row = row+blur_row, neighbour_col= col+blur_col;

                // Boundary Conditions
                if(neighbour_row>=0 && neighbour_row<height && neighbour_col>=0 && neighbour_col<width){
                    int neighbour_idx = neighbour_row * width + neighbour_col;
                    neighbour_sum += input_image[neighbour_idx];
                    ++num_neighbours;
                }
            }
        }

        int idx = row*width+col;
        output_image[idx] = neighbour_sum/num_neighbours;
    }
}


void image_blur(float* input_image_h, float* output_image_h, unsigned int& width, unsigned int& height, unsigned int& blur_size){
    float *input_image_d, *output_image_d;
    unsigned int size = width*height*sizeof(float);

    cudaError_t err ;

    // Allocating CUDA Memory
    err = cudaMalloc((void **)&input_image_d, size);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&output_image_d, size);
    error_check(err, __LINE__, __FILE__);

    // Copying Data From Host To Device
    err = cudaMemcpy(input_image_d, input_image_h, size, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);

    // Calling Kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)std::ceil(width/32.0), (int)std::ceil(height/32.0));
    image_blur_kernel<<<dimGrid,dimBlock>>>(input_image_d, output_image_d, width, height, blur_size);

    err = cudaGetLastError();
    error_check(err, __LINE__, __FILE__);

    // Copying Output from Device to Host
    err = cudaMemcpy(output_image_h, output_image_d, size, cudaMemcpyDeviceToHost);
    error_check(err, __LINE__, __FILE__);

    // Freeing CUDA Memeory
    err = cudaFree(input_image_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(output_image_d);
    error_check(err, __LINE__, __FILE__);
}


int main(){
    unsigned int image_width, image_height, blur_size;

    std::cout<<"Enter the width of the image:";
    std::cin>>image_width;

    std::cout<<"Enter the height of the image:";
    std::cin>>image_height;

    std::cout<<"Enter the blur size:";
    std::cin>>blur_size;

    std::vector<float> input_image(image_width*image_height), output_image(image_width*image_height);

    // Generate Random Numbers
    std::mt19937 gen(666);
    std::uniform_real_distribution<float> N(0.0, 255.0);
    std::generate(std::begin(input_image), std::end(input_image), std::bind(N, gen));

    auto t0 = std::chrono::system_clock::now();

    image_blur(input_image.data(), output_image.data(), image_width, image_height, blur_size);

    auto t1 = std::chrono::system_clock::now();
    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout <<"Elapsed time is:"<<elapsed_par.count() << std::endl;

    return 0;
}