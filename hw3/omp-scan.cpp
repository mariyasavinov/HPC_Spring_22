#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define p 6

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {

	int tid;
	prefix_sum[0] = 0;
	// parallel section
	#pragma omp parallel num_threads(p) private(tid) shared(prefix_sum,A)
	{
	tid = omp_get_thread_num();
	prefix_sum[tid*n/p] = 0;
	for (long i = tid*n/p+1; i < (tid+1)*n/p; i++) { //looping over indices corresponding to that thread
    prefix_sum[i] = prefix_sum[i-1] + A[i-1]; // sum is in same manner as in serial
  }
	}
	// correction
	for (long j = 1; j < p; j++) {
		for (long i = j*n/p; i < (j+1)*n/p; i++) {		
			prefix_sum[i] += prefix_sum[j*n/p-1]+A[j*n/p-1]; 
			}
	}
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);


  free(A);
  free(B0);
  free(B1);
  return 0;
}
