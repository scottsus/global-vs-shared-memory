#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define SIZE 1024

__global__ void matrix_mult(int *a, int *b, int *c)
{
    int my_x = blockIdx.x * blockDim.x + threadIdx.x;
    int my_y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_c = 0;
    for (int i = 0; i < SIZE; i++)
        local_c += a[my_x * SIZE + i] * b[i * SIZE + my_y];
    c[my_x * SIZE + my_y] = local_c;
}

int main()
{
	printf("Un-optimized Matrix Multiplication using Global memory\n");
    int *a = (int *)malloc(sizeof(int) * SIZE * SIZE);
    int *b = (int *)malloc(sizeof(int) * SIZE * SIZE);
    int *c = (int *)malloc(sizeof(int) * SIZE * SIZE);

    for (int i = 0; i < SIZE * SIZE; i++)
    {
        a[i] = 1;
        b[i] = 2;
        c[i] = 0;
    }

    int *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void **)&gpu_a, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void **)&gpu_b, sizeof(int) * SIZE * SIZE);
    cudaMalloc((void **)&gpu_c, sizeof(int) * SIZE * SIZE);
    cudaMemcpy(gpu_a, a, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int) * SIZE * SIZE, cudaMemcpyHostToDevice);

    dim3 dimGrid(64, 64);
    dim3 dimBlock(16, 16);

    struct timespec start, stop;
    double time;

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
        perror("start clock gettime");
    matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
    cudaMemcpy(c, gpu_c, sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
        perror("stop clock gettime");

    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time taken: %f ns\n", time * 1e9);
    printf("c[451][451] = %d\n", c[451 * SIZE + 451]);

    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;
}
