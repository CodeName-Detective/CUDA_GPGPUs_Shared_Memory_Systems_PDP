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