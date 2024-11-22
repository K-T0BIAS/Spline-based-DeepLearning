# About Spline-based-DeepLearning

## New:

**python version of the spline and layer classes**

see [install for python](#install for python) to install

batch compatibility for layers 

**documentation was not yet updated some features might have changed and new features were added**

**updates will follow soon**

## goals:

1. create visual representations for neural networks by replacing commonly used fully connected layers with spline based layers.
2. achieve similar or better precision to common deep learning approaches whilst keeping the structure as light-wheight and fast as possible.
3. allow easy adaptability to existing architectures like convolutional and recurrent networks.

## C++ Implementation/documentation:

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

- single sample forward pass:

**assuming namespace std**
```cpp
vector<double> pred = layer_instance.forward(X, normalize);
```
* vector<double> X = input vector (with size == layer input size)
* bool normalize = output normalization (if True output will be between 0 and 1)
* pred.size() == layer output size

- batched forward pass:
```cpp
vector<vector<double>> pred = layer_instance.forward(X, normalize);
```
* vector<vector<double>> X = batched input (with size == batch size , layer input size)
* bool normalize = output normalization (if True output will be between 0 and 1)
* pred.size() = batch size
* pred[0].size() = layer output size

- single sample backward pass:

**assuming namespace std**
```cpp
vector<double> loss_gradient = layer_instance(X,d_y);
```

* vector<double> X = input (either from previous layer or from dataset)
* vector<double> d_y = loss_gradient (from next layer or loss function)
* loss_gradient == d_y for the previous layers backward pass

- batched backward pass:
```cpp
vector<vector<double>> loss_gradient = layer_instance(X, d_y);
```

* vector<vector<double>> X = batched input (either from previous layer or from dataset)
* vector<vector<double>> d_y = batched loss_gradient (from next layer or from loss function)
* loss_gradient == d_y for the previous layer backward pass (propagated gradient)

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

(when using the manual approach meaning iterating manually over layers to apply activations you have to do the backward pass manually aswell.)

## python Implementation/documentation:

### import

```python
import PySplineNetLib as some_name
```

### Splines
Splines are the main computation unit for this approach, they esentially provide a easily visualizable alterp to wheight matricies

- spline creation:
```python
spline_instance = PySplineNetLib.spline(points,parameters)
```
* points : list = list of points like (num points, 2)
* parameters : list = list of parameters like (num points - 1, 4)

**full example**

```python
points : list = [[0.0,0.0],[0.5,0.25],[1.0,1.0]]
parameters : list = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]

spline_instance = PySplineNetLib.spline(points,parameters)
```

- spline interpolation:

to properly init a spline call .interpolation()

```python
spline_instance.interpolation()
```

this ensures that the parameters are properly set for evaluation and training

- spline forward pass / evaluation:

to evaluate the spline at x call

```python
y : float = spline_instance.forward(x)
```

x : float = point to be evaluated

- spline backward / gradient propagation:

to find the splines gradient based on a give loss grad at spline point (x,y) call

```python
d_y : float = spline_instance.backward(x, d_y, y)
```
x : float = point that was last evaluated
y : float = spline prediction at x
d_y : float = gradient of loss with (x,target) with respect to spline (x,y) (=> loss.backward() or d_y of next layer)

**Note :**

The gradient of this function call is internally stored in the spline.

- adjust spline based on gradient

to apply the gradient from .backward and adjust the spline call:
```python
spline_instance.apply_grad(lr)
```

lr : float = learning rate (controls how strong the gradient affects the splines points)

### layer decumentation comming soon



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
mkdir -p build
cd build
cmake ..
make
cd ..
pip install .
```


## License

This project is licensed under the Mozilla Public License 2.0. 

Copyright (c) 2024 Tobias Karusseit. See the [LICENSE](./LICENSE) file for details.

This project also uses `pybind11`, which is licensed under the MIT License. See [pybind11 GitHub](https://github.com/pybind/pybind11) for more details.

This project also uses `Catch2`, which is licensed under the Boost Software License 1.0. See [Catch2 GitHub](https://github.com/catchorg/Catch2) for more details.