# Testing Execution times using Global vs Shared Memory on the GPU

## Usage
Execute makefile
```
make p1 p2
```
This compiles the .c files using nvcc and sends a batch job using sbatch to CARC, which will assign GPUs to perform the
computationally expensive matrix multiplication.

### Approach 1: Global Memory
Very slow execution time

### Approach 2: Shared Memory
Execution time is faster by orders of magnitude
