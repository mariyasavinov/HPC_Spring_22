/******************************************************************************
* FILE: omp_bug5.c
* DESCRIPTION:
*   Using SECTIONS, two threads initialize their own array and then add
*   it to the other's array, however a deadlock occurs.
* AUTHOR: Blaise Barney  01/29/04
* LAST REVISED: 04/06/05
******************************************************************************/

/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- Added a barrier after the master directive to make the number of threads be
	printed first.
- In the first section, before lockb is set, locka needs to be unset, otherwise
	the issue becomes that locka is set by section 1 and lockb is set by 
	section 2, so there is deadlock.
- In the second section, similarly lockb has to be unset before locka is set.
- Then, in order for a to be added to b before a is changed, both locka AND lockb
	need to be set in each section (in the same order) and both are unset at the end.
	This way one thread adds one vector to the other ENTIRELY before the 2nd thread
	adds them in the opposite manner.
- I.e., either a is added to b entirely  and then b is added to a; or b is added 
	to a entirely and then a is added to b.
- The order of which is added to which first just depends on which thread
	sets a lock first.

******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define PI 3.1415926535
#define DELTA .01415926535

int main (int argc, char *argv[]) 
{
int nthreads, tid, i;
float a[N], b[N];
omp_lock_t locka, lockb;

/* Initialize the locks */
omp_init_lock(&locka);
omp_init_lock(&lockb);

/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
  {

  /* Obtain thread number and number of threads */
  tid = omp_get_thread_num();
  #pragma omp master
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier // added barrier
  printf("Thread %d starting...\n", tid);
  #pragma omp barrier

  #pragma omp sections nowait
    {
    #pragma omp section
      {
      printf("Thread %d initializing a[]\n",tid);
      omp_set_lock(&locka);
      for (i=0; i<N; i++)
        a[i] = i * DELTA;
      omp_unset_lock(&locka); // added
      omp_set_lock(&locka); // added
      omp_set_lock(&lockb);
      printf("Thread %d adding a[] to b[]\n",tid);
      for (i=0; i<N; i++)
        b[i] += a[i];
      omp_unset_lock(&locka);
      omp_unset_lock(&lockb);
      }

    #pragma omp section
      {
      printf("Thread %d initializing b[]\n",tid);
      omp_set_lock(&lockb);
      for (i=0; i<N; i++)
        b[i] = i * PI;
      omp_unset_lock(&lockb); // added
      omp_set_lock(&locka);   
      omp_set_lock(&lockb);   // added
      printf("Thread %d adding b[] to a[]\n",tid);
      for (i=0; i<N; i++)
        a[i] += b[i];
      omp_unset_lock(&locka);
      omp_unset_lock(&lockb);
      }
    }  /* end of sections */
  }  /* end of parallel region */
}


