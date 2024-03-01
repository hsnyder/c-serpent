// Wrapper for cserpent.c to be used as a Python module
// ... Yeah I know, ironic that cserpent can't wrap itself yet
// (at least not with a pleasant python-side API)
//
// Building this will create a python expension module that will
// allow you to call cserpent from python to generate wrapper code.
// The extension module and the standalone executable build are 
// independent. 
//
// You can probably build this with:
//   PY_INC_DIR="$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())")"
//   cc -shared -fPIC -o cserpent.so cserpent_py.c -I$PY_INC_DIR
//
// You'll need the python headers, perhaps from python3-dev or python3-devel
// if you're on a debian or redhat based system, respectively.

#define CSERPENT_SUPPRESS_MAIN
#include "cserpent.c"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>

PyObject* 
wrap_cserpent_main_buffers(PyObject *self, PyObject *args) 
{
	PyObject *argList;
	const char *stdinString;
	if (!PyArg_ParseTuple(args, "Os", &argList, &stdinString)) {
		return 0;
	}

        // make sure we got a list of strings
	if (!PyList_Check(argList)) {
		PyErr_SetString(PyExc_TypeError, "First argument must be a list");
		return 0;
	}

	Py_ssize_t listSize = PyList_Size(argList);
        for(int i = 0; i < listSize; i++) {
                PyObject *item = PyList_GetItem(argList, i);
                if(!PyUnicode_Check(item)) {
                        PyErr_SetString(PyExc_TypeError, "List items must be strings");
                        return 0;
                }
        }

        // Allocate buffers
        size_t bufsz = 4 * 1024 * 1024; // 4MB
	char **cserpent_argv = (char **)malloc((listSize + 1) * sizeof(char *));
	unsigned char *stdout_buf = (unsigned char *)malloc(bufsz);
	unsigned char *stderr_buf = (unsigned char *)malloc(bufsz);

        if(cserpent_argv == 0 || stdout_buf == 0 || stderr_buf == 0) {
                PyErr_NoMemory();
                if(cserpent_argv) free(cserpent_argv);
                if(stdout_buf) free(stdout_buf);
                if(stderr_buf) free(stderr_buf);
                return 0;
        }
        memset(stdout_buf, 0, bufsz);
        memset(stderr_buf, 0, bufsz);

	for (int i = 0; i < listSize; i++) {
		PyObject *item = PyList_GetItem(argList, i);
		cserpent_argv[i] = (char*)PyUnicode_AsUTF8(item); 
	}
	cserpent_argv[listSize] = 0; // Null terminate the array of arguments

	long long stdin_buf_size = strlen(stdinString);
	int retCode = cserpent_main_buffers(
                cserpent_argv, 
                stdin_buf_size, (unsigned char *)stdinString,
                bufsz -1, stdout_buf,
                bufsz -1, stderr_buf);

	PyObject *result = Py_BuildValue("iss", retCode, stdout_buf, stderr_buf);

	free(cserpent_argv);
	free(stdout_buf);
	free(stderr_buf);
	return result;
}

static PyMethodDef module_methods[] = {
	{"cserpent", wrap_cserpent_main_buffers, METH_VARARGS, "Wrap cserpent_main_buffers"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"cserpent",
	NULL,
	-1,
	module_methods
};

PyMODINIT_FUNC PyInit_cserpent(void) {
	return PyModule_Create(&moduledef);
}

