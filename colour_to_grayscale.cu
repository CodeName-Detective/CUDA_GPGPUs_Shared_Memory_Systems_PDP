#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>

#define channels 3

void error_check(cudaError_t& err, int line_no, const char* file_name){
    if (err!=cudaSuccess) {
        std::cout<<cudaGetErrorString(err)<<" at line "<<line_no<<" in file "<<file_name<<std::endl;
    }
}

__global__ void colour_to_gray_kernel(float* color_image, float* grayscale_image, unsigned int width, unsigned int height){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int channel_offset = width * height;

    if (row<height && col<width){
        int idx = row*width+col;

        // grayscale = 0.21*red + 0.72*green + 0.07*blue
        grayscale_image[idx] = (0.21f*color_image[0*channel_offset+idx]) //red channel
                                +(0.72f*color_image[1*channel_offset+idx]) //green channel
                                +(0.07f*color_image[2*channel_offset+idx]); //blue channel
    }
}

void colour_to_gray(float* color_image_h, float* grayscale_image_h, unsigned int& width, unsigned int& height){
    float *color_image_d, *grayscale_image_d;
    unsigned int size = width*height*sizeof(float);

    cudaError_t err ;

    // Allocating CUDA Memory
    err = cudaMalloc((void **)&color_image_d, size*channels);
    error_check(err, __LINE__, __FILE__);
    err = cudaMalloc((void **)&grayscale_image_d, size);
    error_check(err, __LINE__, __FILE__);

    // Copying Data From Host To Device
    err = cudaMemcpy(color_image_d, color_image_h, size*channels, cudaMemcpyHostToDevice);
    error_check(err, __LINE__, __FILE__);

    // Calling Kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)std::ceil(width/32.0), (int)std::ceil(height/32.0));
    colour_to_gray_kernel<<<dimGrid,dimBlock>>>(color_image_d, grayscale_image_d, width, height);

    err = cudaGetLastError();
    error_check(err, __LINE__, __FILE__);

    // Copying Output from Device to Host
    err = cudaMemcpy(grayscale_image_h, grayscale_image_d, size, cudaMemcpyDeviceToHost);
    error_check(err, __LINE__, __FILE__);

    // Freeing CUDA Memeory
    err = cudaFree(color_image_d);
    error_check(err, __LINE__, __FILE__);
    err = cudaFree(grayscale_image_d);
    error_check(err, __LINE__, __FILE__);
}

int main(){
    unsigned int image_width, image_height;

    std::cout<<"Enter the width of the image:";
    std::cin>>image_width;

    std::cout<<"Enter the height of the image:";
    std::cin>>image_height;

    std::vector<float> color_image(image_width*image_height*channels), grayscale_image(image_width*image_height);

    // Generate Random Numbers
    std::mt19937 gen(666);
    std::uniform_real_distribution<float> N(0.0, 255.0);
    std::generate(std::begin(color_image), std::end(color_image), std::bind(N, gen));

    auto t0 = std::chrono::system_clock::now();

    colour_to_gray(color_image.data(), grayscale_image.data(), image_width, image_height);

    auto t1 = std::chrono::system_clock::now();
    auto elapsed_par = std::chrono::duration<double>(t1 - t0);
    std::cout <<"Elapsed time is:"<<elapsed_par.count() << std::endl;

    return 0;
}