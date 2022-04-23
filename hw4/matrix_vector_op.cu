// $ nvcc -arch=sm_61 matrix_vector_op.cu -o matrix_vector_op -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

// CPU vector-vector multiplication
void vec_vec_mult(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    c[i] += a[i]*b[i];
  }
}

// CPU matrix-vector multiplication
void mat_vec_mult(double* c, const double* A, const double* b, long N){
  #pragma omp parallel for num_threads(32)
  for (long i = 0; i < N; i++) {
		for	(long j = 0; j<N; j++) {
    	c[i] += A[i+j*N]*b[j];
		}  
	}
}


// GPU vector-vector multiplication
__global__
void vec_vec_mult_kernel(double* c, const double* a, const double* b, long N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] += a[idx]*b[idx];
}

// GPU matrix-vector multiplication
__global__
void mat_vec_mult_kernel(double* c, const double* A, const double* b, long N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		for (int j = 0; j < N; j++) {
			c[idx] += A[idx+j*N]*b[j];
		}
	}
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  long N = (1UL<<12); //2^12 
	int bl = (1UL<<10);

	// allocate vectors and matrix for matrix-vector product
  double* x = (double*) malloc(N * sizeof(double));
  //double* y = (double*) malloc(N * sizeof(double));  //for vector-vector mult
	double* A = (double*) malloc(N*N * sizeof(double));
	double* z_cpu = (double*) malloc(N * sizeof(double));
	double* z_gpu = (double*) malloc(N * sizeof(double));

	// initialize arrays randomly
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    x[i] = ((double)rand())/RAND_MAX;
    //y[i] = ((double)rand())/RAND_MAX;  //for vector-vector mult
		z_cpu[i] = 0;
		z_gpu[i] = 0;
		for (long j = 0; j < N; j++) {
			A[i+j*N] = ((double)rand())/RAND_MAX;
		}
  }

	// run on CPU with openmp
  double tt = omp_get_wtime();
  //vec_vec_mult(z_cpu, x, y, N);      //for vector-vector mult
	mat_vec_mult(z_cpu, A, x, N);
	tt = omp_get_wtime()-tt;
  printf("\nCPU %f s\n", tt);
	printf("CPU Bandwidth: %f GB/s\n\n", 3*N*N*sizeof(double)/tt/1e9);


	// allocate vectors and matrix on GPU
  double *x_d, *z_d, *A_d; 										//,*y_d; //for vector-vector mult
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  //cudaMalloc(&y_d, N*sizeof(double));		//for vector-vector mult
  cudaMalloc(&z_d, N*sizeof(double));
	cudaMalloc(&A_d, N*N*sizeof(double));
  Check_CUDA_Error("malloc x failed");

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);     //for vector-vector mult
  cudaMemcpy(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z_gpu, N*sizeof(double), cudaMemcpyHostToDevice);
  double ttinner = omp_get_wtime();
  //vec_vec_mult_kernel<<<N/bl,bl>>>(z_d, x_d, y_d, N);    //for vector-vector mult
	mat_vec_mult_kernel<<<N/bl,bl>>>(z_d, A_d, x_d, N);
  cudaDeviceSynchronize();
  ttinner = omp_get_wtime() - ttinner;
  cudaMemcpy(z_gpu, z_d, N*sizeof(double), cudaMemcpyDeviceToHost);
	tt = omp_get_wtime()-tt;
  printf("GPU %f s, %f s\n", tt, ttinner);
	printf("GPU Bandwidth: %f GB/s\n\n", 3*N*N*sizeof(double)/tt/1e9);


  double err = 0;
	// error from the vector vector multiplication
  for (long i = 0; i < N; i++) err += fabs(z_gpu[i]-z_cpu[i]);
  printf("Error = %f\n\n", err);


  cudaFree(x_d);
  //cudaFree(y_d);
  cudaFree(z_d);
	cudaFree(A_d);

  free(x);
  free(A);
	//free(y)
	free(z_cpu);
	free(z_gpu);
  //free(z);
  //free(z_ref);

  return 0;
}

