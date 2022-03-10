/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/

/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- The issue here is in calling a function in the parallel region rather than
	making the reduction sum happen inline. What happens then is that 
	the sum variable is considered, in the function, to be a private
	variable. Reduction clauses cannot be used on private variables, so 
	one fix is to move the entire operation inline. 
- To accomodate the inline change, tid is declared as an integer before the parallel 	
	region, and both i and tid are declared explicitly as private variables 
	in the parallel directive. 

******************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

int tid;

#pragma omp parallel shared(sum) private(i,tid)
  {
  #pragma omp for reduction(+:sum)
    for (i=0; i < VECLEN; i++)
      {
      tid = omp_get_thread_num();
      sum = sum + (a[i]*b[i]);
      printf("  tid= %d i=%d\n",tid,i);
      }
  }


printf("Sum = %f\n",sum);

}
