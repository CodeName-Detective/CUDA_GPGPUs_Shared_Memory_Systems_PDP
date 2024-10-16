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