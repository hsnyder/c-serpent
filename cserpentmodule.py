'''
When developing with Jupyter notebooks, it is nice to be able to 
compile C code into a Python module on the fly, without leaving the
notebook. It's also nice to be able to hot-reload the resulting module.
This module gives you a way to do that. You can write C code in a string,
compile it, wrap it with C-Serpent, and load it as a Python module.

Unfortunately, as I don't think MSVC's cl has a preprocess flag that writes 
to stdout, this will probably only work with gcc or clang. MSVC probably
needs to be treated as a special case in the future.


EXAMPLE USAGE

        m = CSerpentModule('my_module')
        c_code = """
                #include <stdint.h>

                int64_t add_i64(int64_t a, int64_t b) {
                        return a + b;
                }
        """

        m.compile(c_code, ['add_i64'])
        import my_module
        my_module.add_i64(3, 4) # returns 7

        # Now, if you change the C code and recompile, the module will be reloaded
        c_code = """
                #include <stdint.h>

                int64_t add_i64(int64_t a, int64_t b) {
                        return a + b + 1;
                }
        """

        m.compile(c_code, ['add_i64'])
        import my_module
        my_module.add_i64(3, 4) # returns 8

PREREQUISITES

        You need to have built the CSerpent extension module. 
        See comments at the top of cserpent_py.c

'''

import os, sys, time, subprocess, importlib
import cserpent, numpy
from distutils.sysconfig import get_python_inc

compiler_config_gcc_clang = {
        'preprocessor': 'cc -E -', # note the dash: read from stdin
        'preprocessor_include_flag': '-I',
        'compiler': 'cc',
        'include_flag': '-I',
        'linkdir_flag': '-L',
        'rpath_flag': '-Wl,--disable-new-dtags,-rpath,', # set to None if not needed
        'output_flag': '-o ', # trailing space is important
        'default_ccflags': [
                "-Wall", "-Wno-unused-function", # default warnings
                "-shared", "-fPIC", # required for building a shard library
                "-x", "c", "-"] # specify that the input is C code, and read from stdin
}

class CSerpentModule:
        def __init__(self, 
                        module_name, 
                        working_dir='/tmp', 
                        compiler_config=compiler_config_gcc_clang,
                        ):

                if working_dir is None:
                        working_dir = os.getcwd()
                elif working_dir not in sys.path:
                        sys.path.append(working_dir)

                self.modname = module_name
                self.working_dir = working_dir
                self.compiler_config = compiler_config
                self.last_opath = None
        
        def compile(self, c_code, functions, 
                        ccflags=["-O3", "-fopenmp"], 
                        includedirs=[], # recommend absolute paths 
                        linkdirs=[],    # recommend absolute paths 
                        linkflags=[],
                        ):
                
                if 'CSERPENT_EXTRA_INCLUDE_DIRS' in os.environ:
                        includedirs += os.environ['CSERPENT_EXTRA_INCLUDE_DIRS'].split()
                if 'CSERPENT_EXTRA_CCFLAGS' in os.environ:
                        ccflags += os.environ['CSERPENT_EXTRA_CCFLAGS'].split()
                if 'CSERPENT_EXTRA_LINKDIRS' in os.environ:
                        linkdirs += os.environ['CSERPENT_EXTRA_LINKDIRS'].split()
                if 'CSERPENT_EXTRA_LINKFLAGS' in os.environ:
                        linkflags += os.environ['CSERPENT_EXTRA_LINKFLAGS'].split()
                
                includedirs += [get_python_inc(), numpy.get_include()]

                preprocessor_cmd = self.compiler_config['preprocessor'].split()
                preprocessor_cmd += [self.compiler_config['preprocessor_include_flag'] + d for d in includedirs]

                preprocessor_result = subprocess.run(
                        preprocessor_cmd, 
                        input=c_code.encode('utf-8'),
                        stderr=subprocess.PIPE, 
                        stdout=subprocess.PIPE)

                if preprocessor_result.returncode != 0:
                        print("Preprocessing failed:")
                        print(preprocessor_result.stderr.decode('utf-8'))
                        return None
                
                preprocessed_code = preprocessor_result.stdout.decode('utf-8')

                now = str(int(time.time()))
                python_mod_name = self.modname + "_" + now

                cserpent_args = ["-m", python_mod_name, "-D", "-f", "-"] + functions
                cserpent_rtncode, cserpent_stdout, cserpent_stderr = \
                        cserpent.cserpent(cserpent_args, preprocessed_code)
                
                if cserpent_rtncode != 0:
                        print("Error from C-Serpent:")
                        print(cserpent_stderr)
                        return None

                full_code = cserpent_stdout + "\n" + preprocessed_code

                opath=os.path.join(self.working_dir, python_mod_name + ".so")
                compiler_command = self.compiler_config['compiler'].split()
                compiler_command += [self.compiler_config['include_flag'] + d for d in includedirs]
                compiler_command += self.compiler_config['default_ccflags']
                compiler_command += (self.compiler_config['output_flag'] + opath).split()
                compiler_command += ccflags
                compiler_command += [self.compiler_config['linkdir_flag'] + d for d in linkdirs]
                if self.compiler_config['rpath_flag'] is not None:
                        compiler_command += [self.compiler_config['rpath_flag'] + d for d in linkdirs]
                compiler_command += linkflags

                print("Running compiler:")
                print(" ".join(compiler_command))

                compiler_result = subprocess.run(
                        compiler_command, 
                        input=full_code.encode('utf-8'),
                        stderr=subprocess.PIPE)

                if compiler_result.returncode != 0:
                        print("Compilation failed:")
                        print(compiler_result.stderr.decode('utf-8'))
                        return None

                sys.modules[self.modname] = importlib.import_module(python_mod_name)
                if self.last_opath: os.remove(self.last_opath)
                self.last_opath = opath

                print("Build successful!")
                return sys.modules[self.modname]