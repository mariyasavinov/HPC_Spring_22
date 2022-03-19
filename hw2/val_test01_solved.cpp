/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- There is an invalid write because x is an n-sized array with entries 
	x[0], x[1], ..., x[n-1] (so no entry x[n]). As such, the loop over
	i should go from i=2 to i<n
- The delete[] operator is used to destroy objects created with new[], but
	since x was NOT created with new[], we just need to free the allocated
	memory through a "free(x)" command.

******************************************************************************/

# include <cstdlib>
# include <iostream>

using namespace std;

int main ( );
void f ( int n );

//****************************************************************************80

int main ( )

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main program for TEST01.
//
//  Discussion:
//
//    TEST01 calls F, which has a memory "leak".  This memory leak can be
//    detected by VALGRID.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 May 2011
//
{
  int n = 10;

  cout << "\n";
  cout << "TEST01\n";
  cout << "  C++ version.\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  f ( n );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST01\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void f ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    F computes N+1 entries of the Fibonacci sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    18 May 2011
//
{
  int i;
  int *x;

  x = ( int * ) malloc ( n * sizeof ( int ) );

  x[0] = 1;
  cout << "  " << 0 << "  " << x[0] << "\n";

  x[1] = 1;
  cout << "  " << 1 << "  " << x[1] << "\n";

  /*
  for ( i = 2; i <= n; i++ )  --> this results in "Invalid write of size 4"
	because x is an n-size array with entries x[0], x[1], ..., x[n-1]
	i.e. there is no entry x[n]
  Instead, let i=2 to i=n-1:
  */
  for ( i = 2; i < n; i++ ) 
  {
    x[i] = x[i-1] + x[i-2];
    cout << "  " << i << "  " << x[i] << "\n";
  }

  /*
  delete [] x;   --> the delete[] operator is used to destroy objects
	created with new[]  (as well as free the associated allocated memory

  Since x was NOT created with new[], we just need to free the allocated memory,
	which can be done with:
  */
  free(x);

  return;
}

