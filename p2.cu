#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define SIZE 1024
#define BLOCK_SIZE 32

__global__ void matrix_mult(int *a, int *b, int *c)
{
	int row = threadIdx.x, col = threadIdx.y;
	int my_x = blockIdx.x * blockDim.x + row;
	int my_y = blockIdx.y * blockDim.y + col;

	__shared__ int local_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int local_b[BLOCK_SIZE][BLOCK_SIZE];
	int local_c = 0;
	
	for (int i = 0; i < SIZE / BLOCK_SIZE; i++)
	{
		local_a[row][col] = a[my_x * SIZE + (i * blockDim.y + col)];
		local_b[row][col] = b[(i * blockDim.x + row) * SIZE + my_y];
		__syncthreads();
		for (int j = 0; j < BLOCK_SIZE; j++)
			local_c += local_a[row][j] * local_b[j][col];
		__syncthreads();
	}
	c[my_x * SIZE + my_y] = local_c;
}

int main()
{
	printf("Optimized Matrix Multiplication using Shared Memory\n");
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

	dim3 dimGrid(32, 32);
	dim3 dimBlock(32, 32);

	struct timespec start, stop;
	if (clock_gettime(CLOCK_REALTIME, &start) == -1)
		perror("start clock gettime\n");
	matrix_mult<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
	cudaMemcpy(c, gpu_c, sizeof(int) * SIZE * SIZE, cudaMemcpyDeviceToHost);
	if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
		perror("stop clock gettime\n");
	
	double time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;
	printf("Time taken: %f ns \n", time);
	printf("c[451][451] = %d\n", c[451 * SIZE + 451]);

	free(a);
	free(b);
	free(c);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	cudaFree(gpu_c);
	return 0;
	
}
