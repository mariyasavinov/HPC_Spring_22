/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/

/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- There isn't enough memory on stack to have every thread have a private
	version of a, which is a sufficiently large 2D array.
	Instead, I've changed a to a double pointer and at the beginning
	of the parallel section each thread has N^2 size memory allocated
	for the private variable a using malloc. This memory is freed
	at the end of the parallel section.
	This also required slightly changing how a is indexed in the private work.
	This fixes the segmentation fault.
- Also, added barrier directives before and after "Thread X starting.." print
	statement so that the output is cleaner (number of threads prints first,
	then thread starting command).

******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
double *a;

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
  #pragma omp barrier  
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i*N+j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N*N-1]);

  free(a); // free allocated memory
  }  /* All threads join master thread and disband */
  
}




