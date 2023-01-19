WRAPGEN
=======

Wrapgen is a tool designed to make it easier to call C code from Python.
Given a list of files and functions contained within those files, wrapgen
generates code (using the Python C API) that can be compiled into a Python
extension module, and which allows Python code to call the specified C 
functions.

Wrapgen is not intended to be completely general-purpose. In particular:

 - It only supports wrapping a subset of all possible C function signatures. 
   In practice, the limitations are probably acceptable for 95+% of cases.

 - It assumes the use case is scientific or numerical computing, making use
   of numpy.

To elaborate on the second point, the wrappers will only accept numpy arrays
as values for pointer arguments to the underlying function. The exceptions 
to this are `char*` arguments, which are assumed to be strings 
(use `signed char*` for int8), and `void*` which are converted to and from
Python integers. `char*` and `void*` are also supported as return types, but
other pointer types are not (so define your output arrays in Python, and 
populate them with C code). 

For example, consider this C function:

    double mean_i32(int N, int32_t *array) 
    {
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += array[i];
        return sum/N;
    }

The resulting wrapper will accept a Python integer for `N`, and a
numpy array with `dtype=numpy.int32` for `array`. An exception will
be raised if the types don't match.


Usage
-----

Compile wrapgen, then run `wrapgen -h` for detailed usage information.
Typical usage example: 

  $ wrapgen -m coolmodule -f my_c_file.c function1 function2 > wrappers.c   
  $ cc -fPIC -shared -I/path/to/python/headers \
        wrappers.c my_c_file.c \
        -lpython -o coolmodule.so


Compiling
---------

Just point your C compiler at wrapgen.c. For example on a unix-derivative, 

  cc -g wrapgen.c -o wrapgen 
