/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
double *a;//[N][N];

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a) 
  {
  a = (double*) malloc(N*N*sizeof(double)); // not enough memory on stack otherwise

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier  // added this in to make sure number of threads prints first
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i*N+j] = tid + i + j;
  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N*N-1]);

  free(a);
  }  /* All threads join master thread and disband */
  
}




