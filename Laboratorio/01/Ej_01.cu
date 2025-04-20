#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void helloFromGPU() {
    printf("Hola Mundo desde el kernel de CUDA! Hilo: %d\n", threadIdx.x);
}

int main() {
    printf("Hola Mundo desde la CPU!\n");

    helloFromGPU <<< 1, 10 >>> (); 

    cudaDeviceSynchronize(); 
    return 0;
}
