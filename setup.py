from setuptools import setup, Extension
import os
import subprocess
import pybind11

def build_cpp_library():
    """ Build the C++ library using CMake. """
    if not os.path.exists('build'):
        os.makedirs('build')

    # Run cmake to configure the project
    subprocess.check_call(['cmake', '..'], cwd='build')

    # Build the C++ library
    subprocess.check_call(['cmake', '--build', '.'], cwd='build')

def get_library_path():
    """ Get the path to the compiled library. """
    return os.path.join(os.path.abspath('build'), 'lib')

def get_include_path():
    """ Get the path to the include directory. """
    return os.path.abspath('include')

def build_python_extension():
    """ Build the Python extension using setuptools. """
    # Get the relative paths to the library and include directories
    lib_path = os.path.join(os.path.abspath('build'))  # Assuming library is in 'build/lib'
    include_path = os.path.abspath('include')  # Assuming headers are in 'include'

    setup(
        name="PySplineNetLib",  # Name of the generated Python extension module
        version="0.1",
        ext_modules=[
            Extension(
                "PySplineNetLib",  # Name of the generated Python extension module
                ["src/SplineNetLib_py.cpp"],  # Path to your pybind C++ file
                include_dirs=[pybind11.get_include(), include_path],  # Include paths (relative)
                libraries=["SplineNetLib"],  # Link with your precompiled static library (no 'lib' prefix or '.a' extension)
                library_dirs=[lib_path],  # Directory containing the static library (relative path)
                language="c++",  # Ensure it's compiled as C++
            )
        ],
        install_requires=[
            "pybind11>=2.6.0",  # Ensure pybind11 is installed
        ],
    )

def main():
    """ Build the C++ library and then the Python extension. """
    build_cpp_library()
    build_python_extension()

if __name__ == "__main__":
    main()