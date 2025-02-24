﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // OpenMP for CPU parallelism

#define TPB 32

__global__ void blockStripeKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * N + col] = sum;
    }
}

void cpuMatrixMultiply(int* A, int* B, int* C, int N, int startRow, int endRow, int n_threads) {
    omp_set_num_threads(n_threads);
    for (int i = startRow; i < endRow; ++i) {
    #pragma omp parallel for collapse(1)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[j * N + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void transposeMatrix(const int* input, int* output, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            output[j * N + i] = input[i * N + j];
        }
    }
}

void initializeMatrix(int* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = i % 10 + 1;
    }
}

void printMatrix(int* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", mat[i * N + j]);
        }
        printf("\n");
    }
}

void printMatrix(int* mat, int N, int maxSize) {
    if (N > maxSize) {
        printf("Matrix is too large to display completely. Displaying top-left %dx%d submatrix:\n", maxSize, maxSize);
        for (int i = 0; i < maxSize; ++i) {
            for (int j = 0; j < maxSize; ++j) {
                printf("%d ", mat[i * N + j]);
            }
            printf("\n");
        }
    }
    else {
        printf("Full Matrix:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%d ", mat[i * N + j]);
            }
            printf("\n");
        }
    }
}




int main() {
    int n_threads = 12;
    int N = 8192;
    int loadProportion = 512;

    //_sleep(500);

    // Define GPU and CPU work ranges
    int cpuStartRow = 0;
    int cpuEndRow = N / loadProportion; // CPU handles first portion
    int gpuStartRow = N / loadProportion;
    int gpuEndRow = N; // GPU handles the rest

    double cpuStart;
    double cpuEnd;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate host memory
    int* h_A = (int*)malloc(N * N * sizeof(int));
    int* h_B = (int*)malloc(N * N * sizeof(int));
    int* h_C = (int*)malloc(N * N * sizeof(int));
    int* tempMat = (int*)malloc(N * N * sizeof(int));

    initializeMatrix(h_A, N);
    initializeMatrix(tempMat, N);

    transposeMatrix(tempMat, h_B, N);

    // Allocate device memory
    int* d_A, * d_B, * d_C;
    cudaEventRecord(start);
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TPB, TPB);
    dim3 numBlocks((N + TPB - 1) / TPB, (gpuEndRow - gpuStartRow + TPB - 1) / TPB);

    // Prepare CUDA stream for parallel execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    printf("-------------------------\n");
    double startAttempt = omp_get_wtime();

    // Start CPU computation in parallel
    //============================================================
    blockStripeKernel << <numBlocks, threadsPerBlock, 0, stream >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop, stream);

    cpuStart = omp_get_wtime();
    cpuMatrixMultiply(h_A, h_B, h_C, N, cpuStartRow, cpuEndRow, n_threads);
    cpuEnd = omp_get_wtime();
    //============================================================
    cudaMemcpy(h_C + gpuStartRow * N, d_C + gpuStartRow * N, (gpuEndRow - gpuStartRow) * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    double endAttempt = omp_get_wtime();

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);


    printf("CPU Execution Time: %f ms\n", (cpuEnd - cpuStart) * 1000);
    printf("-------------------------\n");
    printf("GPU Execution Time: %f ms\n", gpuTime);
    printf("-------------------------\n");
    printf("Total time: %f ms\n", (endAttempt - startAttempt) * 1000);

    printMatrix(h_C, N, 10);



    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);

    return 0;
}
