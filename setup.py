"""
 Copyright (c) <2024>, <Tobias Karusseit>

 This file is part of the PySplineNetLib project, which is licensed under the 
 Mozilla Public License, Version 2.0 (MPL-2.0).
 
 SPDX-License-Identifier: MPL-2.0

 This file also includes contributions from the pybind11 library, which is licensed 
 under the MIT License.

 SPDX-License-Identifier: MIT

 For the full text of the licenses, see:
 - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0
 - MIT License: https://opensource.org/licenses/MIT
"""



from setuptools import setup, Extension
import os
import subprocess
import pybind11

def build_cpp_library():
    # Run CMake to build the C++ library
    if not os.path.exists('build'):
        os.makedirs('build')

    # Call cmake to configure the project
    subprocess.check_call(['cmake', '..'], cwd='build')

    # Build the C++ library
    subprocess.check_call(['cmake', '--build', '.'], cwd='build')

def get_library_path():
    # Returns the path to the compiled library
    return os.path.join(os.path.abspath('build'))

def get_include_path():
    # Returns the path to the include directory (if needed)
    return os.path.abspath('include')

def build_python_extension():
    # Build the Python extension using setuptools
    setup(
        name="PySplineNetLib",  # Name of the generated Python extension module
        version="0.1",
        ext_modules=[
            Extension(
                "PySplineNetLib",  # Name of the generated Python extension module
                ["src/SplineNetLib_py.cpp"],  # Path to your pybind C++ file
                include_dirs=[pybind11.get_include(), get_include_path()],  # Path to pybind11 and your library's headers
                libraries=["SplineNetLib"],  # Link with your precompiled library
                library_dirs=[get_library_path()],  # Directory containing the precompiled library
                language="c++",  # Ensure it's compiled as C++
            )
        ],
        install_requires=[
            "pybind11>=2.6.0",  # Ensure pybind11 is installed
        ],
    )

def main():
    # Build the C++ library and then the Python bindings
    build_cpp_library()
    build_python_extension()

if __name__ == "__main__":
    main()