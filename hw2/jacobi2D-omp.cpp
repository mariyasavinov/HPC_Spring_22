// Author: Mariya Savinov
// For running this file, the optional inputs are -n, -method, -max_iters, -print_res
// g++ -O3 -std=c++11 -fopenmp jacobi2D-omp.cpp && ./a.out -n 5 -max_iters 30
// g++ -O3 -std=c++11 jacobi2D-omp.cpp && ./a.out -n 5 -max_iters 30


// Necessary packages
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

// Defines a step of the Jacobi algorithm
double Jacobi_step_ij(long N, long i, long j, double *u, double f_ij) { 
	double h = 1.0/(N+1);
	double ustep_ij = h*h*f_ij;  // FIX THIS

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


// Computes the 2-norm residual
double compute_residual(int N, double *u, double *f) {
	double h = 1.0/(N+1);
	double res = 0.;

		// USE OPENMP for THIS!!

	
	for (int i  = 1; i<N+1; i++) {
		for (int j = 1; j<N+1; j++) {
			double Au_ij = 0.;
			// computing matrix-vector product Au
			for (int indx = 0; indx<N+2; indx++) {
				for (int indy = 0; indy<N+2; indy++) {
					if ((abs(i-indx)==1 && abs(j-indy)==0) || (abs(j-indy)==1 && abs(i-indx)==0)) {
						Au_ij += (-1)*u[indx*(N+2)+indy];
					}
				}
			}
			Au_ij += 4*u[i*(N+2)+j];
			Au_ij = Au_ij/(h*h);
			res += (Au_ij - f[i*(N+2)+j])*(Au_ij - f[i*(N+2)+j]);// 2-norm
		}
	}
	
	return sqrt(res);// 2-norm
}		


// Main function
int main(int argc, char** argv) {

	int N = read_option<long>("-n", argc, argv);
		// determines number of discrete points (N+2)x(N+2)
	long max_iters = read_option<long>("-max_iters", argc, argv, "5000");
		// default value is 5000
	long print_res = read_option<long>("-print_res", argc, argv, "0");
		// =1 if you want to print each residual

	// allocate memory for solution u, an (N+2)x(N+2) array of values
	// as well as the step u_step and right-hand-side f
	double* u = (double*) malloc((N+2)*(N+2) * sizeof(double));
	double* u_step = (double*) malloc((N+2)*(N+2) * sizeof(double));
	double* f = (double*) malloc((N+2)*(N+2) * sizeof(double));

	// initialize solution u and step (all zero arrays) and f as = 1
	for (int i = 0; i < N+2; i += 1) {
		for (int j = 0; j<N+2; j += 1) {
			u[i*(N+2)+j] = 0;
			u_step[i*(N+2)+j] = 0;
			f[i*(N+2)+j] = 1;
		}
	}

	Timer t;
	double res = N;
	double ref_res = sqrt(N*N);  // CHECK THIS!!! IS THIS RIGHT?
	long final_iter = max_iters;

	if (print_res==1) {
		printf(" Iter       2-Norm of Residual/reference\n");
	}
	t.tic();
	for (long k = 0; k<max_iters; k+= 1) {
		// Take 1 step of Jacobi
		#pragma omp parallel shared(u, u_step, N)
		{
		#pragma omp for
		for (long i = 1; i < N+1; i++) {
			for (long j = 1; j < N+1; j++) {
				double u_ij = 0;
				u_ij = Jacobi_step_ij(N, i, j, u, f[i*(N+2)+j]);
				u_step[i*(N+2)+j] = u_ij;
			}
		}
		
		// update the solution 
				
		//#pragma omp for  //I think this makes things slower
		for (long i = 0; i<N+2; i++) {
			for (long j = 0; j<N+2; j++) {
				u[i*(N+2)+j] = u_step[i*(N+2)+j];
			}
		}
		}

		
/*
		
		// print output for testing
		printf("k=%d step:\n",k);
		for (long i = 0; i<N+2; i++) {
			printf("%dth row: ",i);
			for (long j = 0; j<N+2; j++) {
				printf("  %.2f  ",u[i*(N+2)+j]);
			}
			printf("\n");
		}
		printf("\n\n");

*/		

		
		// compute residual
		res = compute_residual(N,u,f);

		if (print_res==1) {
			printf("%5d %20e\n",k,res/ref_res);
		}

		// stop iterating if the residual condition is reached
		if (res/ref_res<1e-6) {
			printf("Residual condition reached.\n");
			final_iter = k;
			break;
		}
		

	}
	double time = t.toc();

	//printf("iter %d had residual/reference value = %e\n",final_iter,res/ref_res);


	printf("runtime = %f\n",time);

	//free allocated memory associated with the solution
	free(u);
	free(u_step);

	return 0;
}

