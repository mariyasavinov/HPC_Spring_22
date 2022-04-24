
all: matrix_vector_op jacobi2D-cuda

matrix_vector_op: matrix_vector_op.cu
	nvcc -arch=sm_61 matrix_vector_op.cu -o matrix_vector_op -Xcompiler -fopenmp

jacobi2D-cuda: jacobi2D-cuda.cu
	nvcc -arch=sm_61  jacobi2D-cuda.cu -o jacobi2D-cuda -Xcompiler -fopenmp

clean:
	rm -f jacobi2D-cuda matrix_vector_op

.PHONY: all, clean
