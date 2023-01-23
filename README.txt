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
be raised if the types don't match. The emitted code for this example:

    double  mean_i32 (int  N, int * array);
    PyObject * wrap_mean_i32 (PyObject *self, PyObject *args, PyObject *kwds)
    {
        (void) self;
        static char *kwlist[] = {
            "N",
            "array",0};
        int  N = {0};
        PyArrayObject *array = NULL;
    
        if(!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", kwlist,
            &N,
            &PyArray_Type, &array)) return 0;
    
        if(PyArray_TYPE(array) != C2NPY(int )) {
            PyErr_SetString(PyExc_ValueError, "Invalid array data type for argument 'array' (expected int )");
            return 0;
        }
        if(!PyArray_ISCARRAY(array)) {
            PyErr_SetString(PyExc_ValueError, "Argument 'array' is not C-contiguous");
            return 0;
        }
    
        double  rtn = 0;
        Py_BEGIN_ALLOW_THREADS;
        rtn = mean_i32 (N, PyArray_DATA(array));
        Py_END_ALLOW_THREADS;
        return Py_BuildValue("d", rtn);
    }


Compiling
---------

Just point your C compiler at wrapgen.c. For example on a unix-derivative, 

  $ cc -g wrapgen.c -o wrapgen 
        

Usage
-----

Typical usage example: 

  $ wrapgen -m coolmodule -f my_c_file.c function1 function2 > wrappers.c   
  $ cc -fPIC -shared -I/path/to/python/headers \
       wrappers.c my_c_file.c \
       -lpython -o coolmodule.so

Wrapgen processes its input arguments in-order. First specify the name of the
output python module (which must match the name of the shared library that
you compile) by using the argument sequence '-m modulename'. Then, specify
at least one file, using '-f filename.c'. Then, list the names of the 
functions that you wish to generate wrappers for. You can specify multiple
files like so: '-f minmax.c min max -f avg.c mean median'. The functions are
assumed to be contained in the file specified by the most recent '-f' flag.

Wrapgen invokes the system preprocessor and scans for typedefs in the 
resulting file. It only understands a subset of all possible C typedefs, but
it works for stdint, size_t, and so on. The preprocessor to use is 'cc -E' 
by default, but this can be overridden with the -p flag, or the WRAPGEN_PP
environment variable (the latter takes precedence if both are supplied).

Flags: 
                                                                             
-h   print help message and exit  

-m   the following argument is the name of the module to be built 
     only one module per wrapgen invocation is allowed.
                                                                             
-f   the following argument is a filename.
                                                                             
-v   verbose (prints a list of typedefs that were parsed, for debugging).
                                                                             
-D   disable including declarations for the functions to be wrapped in the 
     generated wrapper file this might be used to facilitate amalgamation 
     builds, for example.
                                                                                                                                                         
-x   if you have some extra handwritten wrappers, you can use '-x whatever'  
     to include the function 'whatever' (calling 'wrap_whatever') in the     
     generated module. You'll need to prepend the necessary code to the file 
     that wrapgen generates.
                                                                             
-p   the following argument specifies the preprocessor to use for future 
     files, if different from the default 'cc -E'. Use quotes if you need
     to include spaces in the preprocessor command.
                                                                             
-P   disable preprocessing of the next file encountered. This flag only lasts 
     until the next file change (i.e. -f).
                                                                             
-i   the following argument is a filename, to be inlcuded before the next  
     file processed (for use with -P).
                                                                             
-I   the following argument is a directory path, to be searched for any  
     future -i flags.
                                                                             
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
                                                                             
WRAPGEN_PP  
     This variable acts like the -p flag (but the -p flag overrides it) 
