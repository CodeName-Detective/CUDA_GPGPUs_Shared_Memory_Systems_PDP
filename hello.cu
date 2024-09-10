#include <stdio.h>

__global__ void hello(){

  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){

  hello<<<2,2>>>();
  cudaDeviceSynchronize();
  /*
  Note the use of cudaDeviceSynchronize() after the kernel launch. 
  In CUDA, kernel launches are asynchronous to the host thread. 
  The host thread will launch a kernel but not wait for it to finish,
   before proceeding with the next line of host code. 
   Therefore, to prevent application termination before the kernel gets to print out its message, 
   we must use this synchronization function.
  */
}

