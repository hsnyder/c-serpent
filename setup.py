from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install
import subprocess
import os

class CustomBuild(build_py):
    def run(self):
        super().run()
        if not self.dry_run:
            setup_dir = os.path.dirname(os.path.abspath(__file__))
            cserpent_path = os.path.join(setup_dir, 'cserpent.c')
            subprocess.check_call(['cc', cserpent_path, '-o', 'cserpent'])
            build_lib = os.path.abspath(self.build_lib)
            self.copy_file('cserpent', build_lib)

setup(
    name='cserpent',
    version='1.1',
    ext_modules=[Extension('cserpent_py', sources=['cserpent_py.c'])],
    cmdclass={ 'build_py': CustomBuild, },
    package_data={'cserpent': ['cserpent']},
    py_modules=['cserpentmodule'],
    install_requires=['numpy']
)

