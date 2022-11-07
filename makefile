CC = nvcc

all: p1 p2

p1: p1.cu
	$(CC) p1.cu -o p1 && sbatch job.sl

p2: p2.cu
	$(CC) p2.cu -o p2 && sbatch job.sl

clean:
	rm p1 p2 gpujob.out