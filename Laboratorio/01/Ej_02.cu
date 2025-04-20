#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <chrono>
#include <random> 

void sumVectors(float* vA, float* vB, float* result, int n){
    for (int i = 0; i < n; i++){
        result[i] = vA[i] + vB[i];
    }
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

    auto start = std::chrono::high_resolution_clock::now();

    sumVectors(vA, vB, result, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Primeros Resultados:\n");
    for (int i = 0; i < 10; i++){
        printf("Elemento %d: %.2f = %.2f + %.2f\n", i, result[i], vA[i], vB[i]);
    }
    printf("Listo. Tiempo de ejecucion en CPU: %f ms\n", duration.count());

    free(vA);
    free(vB);
    free(result);

    return 0;
}