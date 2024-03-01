'''
It's sometimes nice, especially when developing with Jupyter notebooks,
to be able to compile C code into a Python module without leaving the
notebook. It's also nice to be able to hot-reload the resulting module.
This module gives you a way to do that. You can write C code in a string,
compile it, wrap it with C-Serpent, and load it as a Python module.

Unfortunately, as I don't think MSVC's cl has a preproces flag that writes 
to stdout, this will probably only work with gcc or clang.
'''

import os, sys, time, subprocess, importlib
import cserpent, numpy
from distutils.sysconfig import get_python_inc

class CSerpentModule:
        def __init__(self, 
                        module_name, 
                        working_dir='/tmp', 
                        preprocessor='cc -E -',
                        preprocessor_include_flag='-I',
                        compiler='cc', 
                        compiler_include_flag='-I',
                        compiler_linkdir_flag='-L',
                        compiler_rpath_flag='-Wl,--disable-new-dtags,-rpath,', # can set to None if not needed
                        ):

                if working_dir is None:
                        working_dir = os.getcwd()
                elif working_dir not in sys.path:
                        sys.path.append(working_dir)

                self.modname = module_name
                self.working_dir = working_dir
                self.preprocessor = preprocessor
                self.preprocessor_include_flag = preprocessor_include_flag
                self.compiler = compiler
                self.compiler_include_flag = compiler_include_flag
                self.compiler_linkdir_flag = compiler_linkdir_flag
                self.compiler_rpath_flag = compiler_rpath_flag
        
        def compile(self, c_code, functions, 
                        ccflags=["-O3", "-fopenmp"], 
                        includedirs=[], # recommend absolute paths 
                        linkdirs=[],    # recommend absolute paths 
                        linkflags=[],
                        ):
                
                if 'CSERPENT_INCLUDE_DIRS' in os.environ:
                        includedirs += os.environ['CSERPENT_INCLUDE_DIRS'].split()
                if 'CSERPENT_CCFLAGS' in os.environ:
                        ccflags += os.environ['CSERPENT_CCFLAGS'].split()
                if 'CSERPENT_LINKDIRS' in os.environ:
                        linkdirs += os.environ['CSERPENT_LINKDIRS'].split()
                if 'CSERPENT_LINKFLAGS' in os.environ:
                        linkflags += os.environ['CSERPENT_LINKFLAGS'].split()
                
                includedirs += [get_python_inc(), numpy.get_include()]

                preprocessor_cmd = self.preprocessor.split()
                preprocessor_cmd += [self.preprocessor_include_flag + d for d in includedirs]

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
                compiler_command = self.compiler.split()
                compiler_command += [self.compiler_include_flag + d for d in includedirs]
                compiler_command += [
                        "-Wall", "-Wno-unused-function", 
                        "-shared", "-fPIC", 
                        "-x", "c", "-", 
                        "-o", opath]
                compiler_command += ccflags
                compiler_command += [self.compiler_linkdir_flag + d for d in linkdirs]
                if self.compiler_rpath_flag is not None:
                        compiler_command += [self.compiler_rpath_flag + d for d in linkdirs]
                compiler_command += linkflags

                compiler_result = subprocess.run(
                        compiler_command, 
                        input=full_code.encode('utf-8'),
                        stderr=subprocess.PIPE)

                if compiler_result.returncode != 0:
                        print("Compilation failed:")
                        print(compiler_result.stderr.decode('utf-8'))
                        return None

                sys.modules[self.modname] = importlib.import_module(python_mod_name)

                print("Build successful!")
                return sys.modules[self.modname]