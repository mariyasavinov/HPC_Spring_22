// Author: Mariya Savinov
// For running this file, the optional inputs are -n, -method, -max_iters, -print_res
// g++ -O3 -std=c++11 -fopenmp jacobi2D-omp.cpp && ./a.out -n 5 -max_iters 30
// g++ -O3 -std=c++11 jacobi2D-omp.cpp && ./a.out -n 5 -max_iters 30


// Necessary packages
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>

// Defines a step of the Jacobi algorithm using CPU
double Jacobi_step_ij(long N, long i, long j, double *u, double f_ij) { 
	double h = 1.0/(N+1);
	double ustep_ij = h*h*f_ij; 

	for (int indx = 0; indx<N+2; indx++) {
		for (int indy = 0; indy<N+2; indy++) {
			if ((abs(i-indx)==1 && abs(j-indy)==0) || (abs(j-indy)==1 && abs(i-indx)==0)) {
				ustep_ij += u[indx*(N+2)+indy];
			}
		}
	}
	
	ustep_ij = ustep_ij/4;

	return ustep_ij;
}

// Defines a step of the Jacobi algorithm using GPU
__global__
void Jacobi_step_kernel(long N, double *u, double *u_step, double *f) { 
	double h = 1.0/(N+1);
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if (i < N+1) {
		for (long j = 1; j < N+1; j++) {
			u_step[i*(N+2)+j] = (h*h*f[i*(N+2)+j]+u[i*(N+2)+j+1]+u[i*(N+2)+j-1] + u[(i+1)*(N+2)+j] + u[(i-1)*(N+2)+j])/4;
			/*
			u_step[i*(N+2)+j] = h*h*f[i*(N+2)+j]/4;
			for (int indx = 0; indx<N+2; indx++) {
				for (int indy = 0; indy<N+2; indy++) {
					if ((abs(i-indx)==1 && abs(j-indy)==0) || (abs(j-indy)==1 && abs(i-indx)==0)) {
						u_step[i*(N+2)+j] += u[indx*(N+2)+indy]/4;
					}
				}
			}
			*/
		}
	}

}

// Defines the update step of the algorithm using GPU
__global__
void update_soln_kernel(long N, double *u, double *u_step) { 
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	
	if (i < N+1) {
		for (long j = 1; j<N+1; j++) {
			u[i*(N+2)+j] = u_step[i*(N+2)+j];
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


// Main function
int main(int argc, char** argv) {

	int N = 64;
	int bl = 32;
	int max_iters = 40;
	int threadnum = 6;

	// allocate memory for solution u (CPU and GPU), (N+2)x(N+2) arrays of values
	// as well as the step u_step and right-hand-side f
	double* u_cpu = (double*) malloc((N+2)*(N+2) * sizeof(double));
	double* u_gpu = (double*) malloc((N+2)*(N+2) * sizeof(double));
	double* u_step = (double*) malloc((N+2)*(N+2) * sizeof(double));
	double* f = (double*) malloc((N+2)*(N+2) * sizeof(double));

	// initialize solutions and step (all zero arrays) and f as = 1
	for (int i = 0; i < N+2; i += 1) {
		for (int j = 0; j<N+2; j += 1) {
			u_cpu[i*(N+2)+j] = 0;
			u_gpu[i*(N+2)+j] = 0;
			u_step[i*(N+2)+j] = 0;
			f[i*(N+2)+j] = 1;
		}
	}

	double tt = omp_get_wtime();

	printf("\nFor (N+2)x(N+2) grid with N=%d, %d threads, and Dirichlet boundary conditions u = 0:\n\n",N,threadnum); 
	printf("Using CPU with OpenMP:\n");

	for (int k = 0; k<max_iters; k+= 1) {
		// Take 1 step of Jacobi
		#pragma omp parallel shared(u_cpu, u_step, N) num_threads(threadnum)
		{
		#pragma omp for
		for (long i = 1; i < N+1; i++) {
			for (long j = 1; j < N+1; j++) {
				double u_ij = 0;
				u_ij = Jacobi_step_ij(N, i, j, u_cpu, f[i*(N+2)+j]);
				u_step[i*(N+2)+j] = u_ij;
			}
		}
		}
		// update the solution 
		for (long i = 0; i<N+2; i++) {
			for (long j = 0; j<N+2; j++) {
				u_cpu[i*(N+2)+j] = u_step[i*(N+2)+j];
			}
		}	

	}
	tt = omp_get_wtime() -tt;

	printf("CPU %f s\n\n",tt);

	for (int i = 0; i < N+2; i += 1) {
		for (int j = 0; j<N+2; j += 1) {
			u_step[i*(N+2)+j] = 0;
		}
	}

	//---------------------------------------------------------------------

	// allocate memory on GPU
	double *u_step_d, *u_d, *f_d;
	cudaMalloc(&u_d, (N+2)*(N+2)*sizeof(double));
	Check_CUDA_Error("malloc u failed");
	cudaMalloc(&u_step_d, (N+2)*(N+2)*sizeof(double));
	cudaMalloc(&f_d, (N+2)*(N+2)*sizeof(double));

	tt = omp_get_wtime();
	cudaMemcpy(u_d, u_gpu, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(u_step_d, u_step, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(f_d, f, (N+2)*(N+2)*sizeof(double), cudaMemcpyHostToDevice);
	double ttinner = omp_get_wtime();
	

	printf("Using GPU with cuda:\n\n");


	for (int k = 0; k<max_iters; k += 1) {
		// Take 1 step of Jacobi
		Jacobi_step_kernel<<<N/bl,bl>>>(N, u_d, u_step_d, f_d);
		cudaDeviceSynchronize();

		// update the solution
		update_soln_kernel<<<N/bl,bl>>>(N, u_d, u_step_d);
		cudaDeviceSynchronize();

	}
	ttinner = omp_get_wtime() - ttinner;

	// copy back to CPU
	cudaMemcpy(u_gpu, u_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);

	tt = omp_get_wtime() - tt;

	printf("GPU %f s, %f s\n", tt, ttinner);

	// check error between GPU and CPU results
	double err = 0;	
	for (long i = 0; i < N+2; i++) {
		for (long j = 0; j < N+2; j++) {
			err += fabs(u_cpu[i*(N+2)+j]-u_gpu[i*(N+2)+j]);
		}
	}
	printf("Error = %f\n\n",err);
	/*
	for (long i = 40; i<44; i++) {
		printf("%d entry of cpu: %e\n",i, u_cpu[i*(N+2)]);
		printf("%d entry of gpu: %e\n",i, u_gpu[i*(N+2)]);
	}
	*/

	//free allocated memory associated with the solution
	free(u_cpu);
	free(u_gpu);
	free(u_step);
	free(f);

	cudaFree(f_d);
	cudaFree(u_d);
	cudaFree(u_step_d);

	return 0;
}

