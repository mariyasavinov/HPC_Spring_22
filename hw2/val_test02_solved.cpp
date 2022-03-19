/******************************************************************************
Comments on what was wrong: [Mariya Savinov]
- Originally only the first 5 entries of x are initialized, but
	the remaining lines of the program require the use of the remaining
	entries e.g. to set x[2], x[5] as well as when x[i]=2x[i]
- This resuts in a valgrind error of ``Conditional jump or move depends
	on uninitialised value(s)" and ``Use of uninitialised value"
- So, I assume that the intention is for the remainder of uninitialized
	entries to be initialized as x[i] = i as well,
	ultimately resulting in a 
	print statement where x = [0 2 14 6 8 12 12 14 16 18]
******************************************************************************/


# include <cstdlib>
# include <iostream>

using namespace std;

void junk_data ( );
int main ( );

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
//    TEST02 has some uninitialized data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    19 May 2011
//
{
  cout << "\n";
  cout << "TEST02:\n";
  cout << "  C++ version\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  junk_data ( );
//
//  Terminate.
//
  cout << "\n";
  cout << "TEST02\n";
  cout << "  Normal end of execution.\n";

  return 0;
}
//****************************************************************************80

void junk_data ( )

//****************************************************************************80
//
//  Purpose:
//
//    JUNK_DATA has some uninitialized variables.
//
//  Discussion:
//
//    VALGRIND's MEMCHECK program monitors uninitialized variables, but does
//    not complain unless such a variable is used in a way that means its
//    value affects the program's results, that is, the value is printed,
//    or computed with.  Simply copying the unitialized data to another variable
//    is of no concern.
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

  x = new int[10];
//
//  X = { 0, 1, 2, 3, 4, ?a, ?b, ?c, ?d, ?e }.
//
  for ( i = 0; i < 5; i++ )
  {
    x[i] = i;
  }

  // Initialize remainder of array
  for ( i = 5; i<10; i++ )
  {
    x[i] = i;
  }

//
//  Copy some values.
//  X = { 0, 1, ?c, 3, 4, ?b, ?b, ?c, ?d, ?e }.
//
  x[2] = x[7];
  x[5] = x[6];
//
//  Modify some uninitialized entries.
//  Memcheck doesn't seem to care about this.
//
  for ( i = 0; i < 10; i++ )
  {
    x[i] = 2 * x[i];
  }
//
//  Print X.
//
  for ( i = 0; i < 10; i++ )
  {
    cout << "  " << i << "  " << x[i] << "\n";
  }

  delete [] x;

  return;
}

