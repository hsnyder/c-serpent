<img src='https://github.com/hsnyder/c-serpent/blob/master/logo.png' height=128 width=128 />

C-SERPENT
=========

C-serpent is a tool designed to make it easier to call C code from Python.
CPython (the main Python implementation) has a C API that makes it possible
to write "extension modles" in C. These extension modules are imported and
called just like regular Python code, but are actually written in C.
However, that API has a bit of a learning curve. C-serpent aims to make 
this process easier by automatically generating the necessary "wrapper" 
code that is required to make a C function callable just like an ordinary
python function. 

C-serpent is a standalone program. Given a list of file paths and function 
names, c-serpent reads the specified files and parses the type signatures 
of the specified functions, then generates wrapper functions which use the
Python C API to do the necessary conversions between Python datatypes and
C datatypes, and so on. 

C-serpent is not completely general-purpose:

 - It only supports wrapping a subset of all possible C function signatures 
   (details below).

 - It assumes the use of the popular `numpy` package for numerical arrays. 


| C type                     | Python type                |
| -------------------------- | -------------------------- |
| signed/unsigned char       | int                        |
| signed/unsigned short      | int                        |
| signed/unsigned int        | int                        |
| signed/unsigned long       | int                        |
| signed/unsigned long long  | int                        |
| float                      | float                      |
| double                     | float                      | 
| long double                | (not supported)            |
| complex float              | (not supported)            |
| complex double             | (not supported)            |
| complex long double        | (not supported)            |
| char*                      | str                        |
| void*                      | int                        |
| signed/unsigned char*      | numpy array (int8/uint8)   |
| signed/unsigned short*     | numpy array (int16/uint16) |
| signed/unsigned int*       | numpy array (int32/uint32) |
| signed/unsigned long*      | numpy array (varies)       |
| signed/unsigned long long* | numpy array (int64/uint64) |

`long*` arguments are mapped either to 32 or 64 bit integers, depending
on the size of `long` on your platform (use `int32_t`/`int64_t` from 
`stdint.h` to avoid this). `char*` arguments are assumed to be strings 
(use `signed char*` for int8). `void*` arguments are converted to and 
from Python integers. `char*` and `void*` are also supported as return 
types, but other pointer types are not (so define your output arrays in 
Python, and populate them with C code). Non-void pointer arguments also 
accept `None`, which results in a null pointer being passed to the C function.
`stdint.h` types are supported, since they are just typedefs.

Example, consider this C function:

    double mean_i32(int N, int32_t *array) 
    {
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += array[i];
        return sum/N;
    }

The resulting wrapper will accept a Python integer for `N`, and either `None` 
or a numpy array with `dtype=numpy.int32` for `array`. An exception will
be raised if the supplied types don't match. 

Compiling
---------

To built C-serpent, just point your C compiler at c-serpent.c. 
For example on a unix-derivative, 

    $ cc -g c-serpent.c -o cserpent 

If you can define `CSERPENT_DISABLE_ASSERT` to compile out assertions, if
you wish to.
        
Usage
-----

C-serpent generates code, so you'll have to compile that code into an extension
module in order to actually import and use the module. 

Consider the same example function from above:

    double mean_i32(int N, int32_t *array) 
    {
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += array[i];
        return sum/N;
    }

To make this function callable from Python, you might do the following:

1. save that function to a file (e.g. `mean.c`)

2. ask python where its headers are

     $ python
     >>> from distutils.sysconfig import get_python_inc
     >>> get_python_inc()
     '/usr/include/python3.11'
     >>> import numpy
     >>> numpy.get_include()
     '/usr/lib/python3.11/site-packages/numpy/core/include'

3. use c-serpent to generate the wrapper code

     $ c-serpent -m means -f mean.c mean_i32 > mean_wrappers.c   

4. compile the wrapper code and the original C code into an extension module

     $ cc -fPIC -shared \
          -I/usr/lib/python3.11/site-packages/numpy/core/include \
          -I/usr/include/python3.11 \
          mean_wrappers.c mean.c \
          -lpython -o means.so

5. call it:
 
     $ python
     >>> import means, numpy
     >>> x = numpy.array([1,2,3,4], dtype=numpy.int32)
     >>> means.mean_i32(len(x), x)
     2.5

C-serpent processes its command-line arguments in order. 
First specify the name of the output python module (which must match the name 
of the shared library that you plan to compile) by using the argument sequence 
'-m modulename'. Then, specify a file, using '-f filename.c'.  Then, list the 
names of the functions within that file that you wish to generate wrappers for. 
You can specify multiple files like so: 
'-f minmax.c min max -f avg.c mean median'. 

C-serpent invokes the system preprocessor and scans for typedefs in the 
resulting file. It only understands a subset of all possible C typedefs, but
it works for `stdint.h`, `size_t`, and so on. The preprocessor used is 'cc -E' 
by default, but this can be overridden with the -p flag, or the `CSERPENT_PP`
environment variable (the former takes precedence if both are supplied).

Flags:   
                                                                               
    -h   print help message and exit    
      
    -m   the following argument is the name of the module to be built   
         only one module per c-serpent invocation is allowed.  
                                                                                   
    -f   the following argument is a filename.  
                                                                                   
    -v   verbose (prints a list of typedefs that were parsed, for debugging).  
                                                                                   
    -D   by default, the output wrapper file contains declarations for the functions
         to be wrapped. This is useful if you don't want to write a separate `.h`
         file, but can be inconvenient if you're including all source files in one
         translation unit. The `-D` flag suppresses these declarations.

    -x   if you are writing some wrappers by hand (e.g. because C-serpent doesn't
         support a particular type or usage pattern), you can use '-x whatever'    
         to include the function 'whatever' in the generated module definition. 
         The function name as seen by Python will be `whatever`, and you'll need to
         supply a function called `wrap_whatever`, which will need to be prepended
         to the code that C-serpent generates. 
                                                                                   
    -p   the following argument specifies the preprocessor to use for future   
         files, if different from the default 'cc -E'. Use quotes if you need  
         to include spaces in the preprocessor command.  
                                                                                   
    -P   disable preprocessing of the next file encountered. This flag only lasts   
         until the next file change (i.e. -f).  
                                                                                   
    -t   the following argument is a type name, which should be treated as 
         being equivalent to void. This is useful for making c-serpent handle
         pointers to unsupported types (e.g. structs) as void pointers (thereby
         converting them to and from python integers). 

         this flag only lasts until the next file change (i.e. -f)   
                                                                                   
    -i   the following argument is a filename, to be included before the next    
         file processed (for use with -P).  

    -I   the following argument is a directory path, to be searched for any    
         future -i flags.  
                                                                                   
    -g   functions that follow are "generic". This is explained fully below.      
                                                                                   
         this flag only lasts until the next file change (i.e. -f)   
                                                                                    
    -G   By default, when processing generic functions, c-serpent will remove a  
         trailing underscore from the names of the generated dispatcher function 
         (e.g. for functions sum_f and sum_d, the arguments -g -G sum_ would result
         in the dispatcher function simply being called sum). This flag disables 
         that functionality, causing trailing underscores to be kept.            
                                                                                   
         this flag only lasts until the next file change (i.e. -f)                                                                                
    -e   for functions that follow: if they return a string (const char *), the    
         string is to be interpreted as an error message (if not null) and a python  
         exception should be thrown.  
                                                                                   
         this flag only lasts until the next file change (i.e. -f)   
                                                                                   
    -e,n,chkfn   for functions that follow: after calling, another function called  
         chkfn should be called.  chkfn should have the signature    
         'const char * checkfn (?)' where ? is the type of the n-th argument to the  
         function (0 means the function's return value). if the chkfn call returns  
         a non-null string, that string is assumed to be an error message and a    
         python exception is generated.   
                                                                                   
         this flag only lasts until the next file change (i.e. -f)   
                                                                               
Environment variables:   
                                                                               
CSERPENT_PP    
     This variable acts like the -p flag (but the -p flag overrides it)   
                                                                               
Generic functions:   
                                                                               
     If you have several copies of a function that accept arguments that are   
     of different data types, then c-serpent may be able to automatically      
     generate a disapatch function for you, that allows it to be called from   
     python in a type-generic way. In order to use this feature, your function 
     must use a function-name suffix to indicate the data type, following this 
     convention: 
                                                                               
       type            suffix 
       ----            ------ 
       int8            b 
       int16           s 
       int32           i 
       int64           l 
        
       uint8           B 
       uint16          S 
       uint32          I 
       uint64          L 
        
       float           f 
       double          d 
        
       complex float   F 
       complex double  D 
                                                                               
     You do not need to supply all of these variants; c-serpent will support   
     whichever variants it finds. 
                                                                               
     Example: consider: $ ./c-serpent -m mymodule -f whatever.c -g mean        
     If whatever.c contains the following functions, then python code will be  
     able to call `mymodule.mean(N, arr)` where arr is a float or double array 
                                                                               
       double meanf(int N, float *arr);                                        
       double meand(int N, double *arr);                                       
                                                                               
     C-serpent will try to figure out which arguments change according to the  
     convention and which do not. Return values may also change.               
                                                                               
     Lastly, the type-specific versions of the function do still get wrapped.
