#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <chrono>
#include <random> 

__global__ void sumVectorsKernel(float* vA, float* vB, float* result, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) result[i] = vA[i] + vB[i];
}

int main(){
    int n = 1000000;
    size_t size = n * sizeof(float);

    float* vA = (float*)malloc(size);
    float* vB = (float*)malloc(size);
    float* result = (float*)malloc(size);

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

    for (int i = 0; i < n; i++){
        vA[i] = distribution(generator);
        vB[i] = distribution(generator);
    }

    float* d_vA, * d_vB, * d_result;
    cudaMalloc((void**)&d_vA, size);
    cudaMalloc((void**)&d_vB, size);
    cudaMalloc((void**)&d_result, size);

    cudaMemcpy(d_vA, vA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vB, vB, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);

    sumVectorsKernel <<<numBlocks, blockSize >>> (d_vA, d_vB, d_result, n);

    cudaEventRecord(stop);

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);

    bool verification = true;
    for (int i = 0; i < 10; i++)
        if (result[i] != vA[i] + vB[i]){
            printf("Error en Elemento[%d] = %f\n", i, result[i]);
            verification = false;
            break;
        }

    if (verification){
        printf("Listo\n");
    }

    printf("Tiempo de ejecucion del kernel en la GPU: %f ms\n", gpu_milliseconds);
    printf("Tiempo total de ejecucion en la GPU: %f ms\n", cpu_duration.count());

    cudaFree(d_vA);
    cudaFree(d_vB);
    cudaFree(d_result);

    free(vA);
    free(vB);
    free(result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}