# Makefile for easy install and builds 

default: build_ext

build_ext: setup.py src/rad_hydro.pyx src/rad.cpp src/rad.hpp
	CC=gcc-10 python setup.py build_ext --inplace && rm -f src/rad_hydro.cpp && rm -Rf build
	
install: setup.py src/rad_hydro.pyx src/rad.cpp src/rad.hpp
	pip install .  && rm -f src/rad_hydro.cpp && rm -Rf build
install_dev: setup.py src/rad_hydro.pyx src/rad.cpp src/rad.hpp
	pip install -e .  && rm -f src/rad_hydro.cpp && rm -Rf build

clean:
	rm *.so