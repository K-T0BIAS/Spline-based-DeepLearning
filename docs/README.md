# About Spline-based-DeepLearning

## New:
batch compatibility for layers 

**documentation was not yet updated some features might have changed and new features were added**

**updates will follow soon**

## goals:

1. create visual representations for neural networks by replacing commonly used fully connected layers with spline based layers.
2. achieve similar or better precision to common deep learning approaches whilst keeping the structure as light-wheight and fast as possible.
3. allow easy adaptability to existing architectures like convolutional and recurrent networks.

## Implementation:

### Splines:
The splines are the main computation unit of a layer. They allow for an easily adjustable and visualizable alternative to wheight matricies.
To create a spline call:
```cpp
SplineNetLib::spline spline_instance = spline(points,parameters);
```
where points and parameters are vectors of shapes:

$$
( \text{output size}, \text{input size}, 2)
$$

and

$$
( \text{output size},\text{input size}, 4)
$$

**Note** that the x values of the points list must be sorted from smallest to biggest.

to fully initialize the spline call:
```cpp
Spline_instance.interpolate();
```
this, although not always nessecery  will adjust the parameters with respect to the points.

To evaluate the spline at point x do:
```cpp
double y = Spline_instance.forward(x)
```
**Note** that x must be between 0 and the largest x value in the splines points list. Trying to access x values outside the spline will result in an error.

To perform a backward pass call:
```cpp
double loss_grad = spline.backward(x,d_y,lr)
```
* double x = input
* double d_y = loss Gradient of the next layer
* double lr = learning rate

### layers
A layer uses splines as substitution for wheight and bias matricies.
Layers are implemented similar to torch.nn.linear();
To create a new layer call:
```cpp
SplineNetLib::layer layer_instance = layer(in_size,out_size,detail,max);
```
* unsigned int in_size = num of elements in the input vector
* unsigned int out_size = num of elements in the target vector (like neurons in linear)
* unsigned int detail = num of controlpoints (exept for default points at 0,0 and max,0)
* double max = Maximum x value (recomended to be 1.0)

To load a layer from previously found points call:
```cpp
SplineNetLib::layer layer_instance = layer(points,parameters);
```
**assuming namespace std**
* vector<vector<vector<vector<double>>>> points ({{{{x,y},...},...},...}) = Matrix like (input size • output size • detail + 2 • 2)
* vector<vector<vector<vector<double>>>> parameters ({{{{0,0,0,0},...},...},...} = Matrix like (input size • output size • detail + 1 • 4)

To fully init a layer call:
```cpp
layer_instance.interpolate_splines();
```
**Single layer training:**

- forward pass:
```cpp
vector<double> Y = layers_instance.forward(X,normalize);
```
* vector<double> X = input vector (with size == layers input size)
* bool normalize = output normalization (if True output will be between 0 and 1)
* Y.size()-1 == layer output size

- backward pass:
```cpp
vector<double> loss_gradient = layer_instance(X,d_y);
```

* vector<double> X = input (either from previous layer or from dataset)
* vector<double> d_y = loss_gradient (from next layer or loss function)
* loss_gradient == d_y for the previous layers backward pass

**layer size:**

$$
\text{layer parameters} = \text{input size} × \text{output size} × (\text{detail} + 2) × 2 + \text{input size} * \text{output size} × (\text{detail} + 1) × 4
$$

### Network

To create a spline network call
```cpp
SplineNetLib::nn network_instance = nn(num_layers,input_sizes,output_sizes,details,max_values)
```
**assuming namespace std**
* int num_layers = number of layers the network is supposed to have
* vector<unsigned int> input_sizes = input_sizes for the layer at each index (e.g. {2,3} layer 0 takes 2 inputs)
* vector<unsigned int> output_sizes = output_sizes for each layer
* vector<double> details = detail for each layer
* vector<double> max_values = max value for each layer (best to set all layers except last to 1.0 and use activation functions to normalize the output between 0 and 1)

**Training**

- forward pass:

  ```cpp
  std::vector<double> pred = network_instance.forward(X, normalize)
  ```
  * vector<double> X = input
  * bool normalize = normalize outputs (not recommended better use activation functions and itterate manually over the layers)
 
- backwards pass

```cpp
std::vector<double> loss_gradient = network_instance.backward(X,d_y)
```
* std::vector<double> X = forward prediction
* std::vector<double> d_y = loss_gradient

(when using the manual approach meaning iterating manually over layers to apply activations you have to do the backward pass manually aswell. In the future we hope to include a activation function pointer to take care of handling activations in layers directly)


## install for c++

1. git clone https://github.com/K-T0BIAS/Spline-based-DeepLearning.git
2. cd Spline-based-DeepLearning
3. mkdir build
4. cd build
5. cmake ..
6. make
7. make install or make install DESTDIR=/path_to_desired_directory

8. to run the example : ./SplineNetExample

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

1. git clone https://github.com/K-T0BIAS/Spline-based-DeepLearning.git
2. cd Spline-based-DeepLearning
3. mkdir -p build
4. cd build
5. cmake ..
6. make
7. cd ..
8. pip install .



## License

This project is licensed under the Mozilla Public License 2.0. 

Copyright (c) 2024 Tobias Karusseit. See the [LICENSE](./LICENSE) file for details.

This project also uses `pybind11`, which is licensed under the MIT License. See [pybind11 GitHub](https://github.com/pybind/pybind11) for more details.

This project also uses `Catch2`, which is licensed under the Boost Software License 1.0. See [Catch2 GitHub](https://github.com/catchorg/Catch2) for more details.