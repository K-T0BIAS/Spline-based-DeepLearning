#Due to a personal lack of time and resources, work on this project was stopped. Please be aware that some features may still be incomplete or faulty. Should questions arise please feel free to contact me at any time.

# About Spline-based-DeepLearning

## bugs

* reshaping a CTensor after performing operations on it may break the gradient calculation!

## Table of contents

[goals](#goals)

[C++ spline documentation](docs/cpp_splines.md)

[C++ CTensor documentation](docs/cpp_CTensor.md)

[python spline documentation](docs/py_splines.md)


## New:

* python version of the spline and layer classes

see [install for python](#install-for-python) to install

* batch compatibility for layers 

* CTensor class (tensor class with automatic computation graph and gradient propagation)

* python version for CTensor

**documentation was not yet updated some features might have changed and new features were added**

**updates will follow soon**

## goals

1. create visual representations for neural networks by replacing commonly used fully connected layers with spline based layers.
2. achieve similar or better precision to common deep learning approaches whilst keeping the structure as light-wheight and fast as possible.
3. allow easy adaptability to existing architectures like convolutional and recurrent networks.

## install for c++

```txt
git clone https://github.com/K-T0BIAS/Spline-based-DeepLearning.git
cd Spline-based-DeepLearning
mkdir build
cd build
cmake ..
make
make install or make install DESTDIR=/path_to_desired_directory
```
to run the example : ./SplineNetExample

## include

in .cpp:
```cpp
#include "SplineNetLib/Network.hpp"
```

in the projects cmake:
```txt
cmake_minimum_required(VERSION 3.10)
project(YourProjectDirectory)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find SplineNetLib package
find_package(SplineNetLib REQUIRED)

# Add executable and link with SplineNetLib
add_executable(YourProjectDirectory main.cpp)
target_link_libraries(YourProjectDirectory PRIVATE SplineNetLib)
```
 
or in terminal:
```txt
g++ -std=c++17 -I/path_to_include -L/path_to_lib -lSplineNetLib main.cpp -o YourProjectDirectory 
```
* Replace /path_to_include with the path to the installed include directory.

* Replace /path_to_lib with the path where libSplineNetLib.a is located.

## install for python

**Note this only includes splines and layer, no Network class**

**REQUIRED: pybind11, setuptools, wheel (if not already install these with pip)**

```txt
git clone https://github.com/K-T0BIAS/Spline-based-DeepLearning.git
cd Spline-based-DeepLearning
pip install .
```


## License

This project is licensed under the Mozilla Public License 2.0. 

Copyright (c) 2024 Tobias Karusseit. See the [LICENSE](./LICENSE) file for details.

This project also uses `pybind11`, which is licensed under the MIT License. See [pybind11 GitHub](https://github.com/pybind/pybind11) for more details.

This project also uses `Catch2`, which is licensed under the Boost Software License 1.0. See [Catch2 GitHub](https://github.com/catchorg/Catch2) for more details.
