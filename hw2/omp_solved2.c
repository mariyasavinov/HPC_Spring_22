/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/

/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- Before the fix, tid is shared by all threads, preventing the printf statement
	at the end to differentiate between the threads that run it because
	tid is set and shared among all threads. Adding "private(tid)" clause
	to the beginning of the parallel region fixes this
- To enforce that the printf statement of the number of threads is performed
	by only the master thread, change the if statement to a master directive
	("pragma omp master")
- Assuming the number of threads should be printed before "Thread x starting" is 
	printed, added another barrier after the master directive.
- The parallel for loop needs a reduction sum for total, otherwise the resultant
	total is from a single thread's output.
	Note that when running this code, the result varies around 5e11, but
	this is an issue of floating point arithmetic since the total
	is not added up the same way every time.

******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total;

/*** Spawn parallel region ***/
#pragma omp parallel private(tid)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  #pragma omp master 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  total = 0.0;
  #pragma omp for schedule(dynamic,10) reduction(+: total)
  for (i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}

