import numpy
import os 
from setuptools import setup, setuptools, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils import sysconfig

def get_ext_filename_without_platform_suffix(filename):
    name, ext  = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext

class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)
    
extensions = [
    Extension("rad_hydro", 
        sources=["src/rad_hydro.pyx", "src/rad.cpp", "units_lib/units.cpp"],
        include_dirs=[numpy.get_include(), "units_lib"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]
setup(
    name='sogbo',
    version='0.0.1',
    author='Marcus DuPont',
    packages=['sogbo'],
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix},
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)