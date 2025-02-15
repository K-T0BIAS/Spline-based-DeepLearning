### splines

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
vector<double> loss_gradient = layer_instance.backward(X,d_y);
```

* vector<double> X = input (either from previous layer or from dataset)
* vector<double> d_y = loss_gradient (from next layer or loss function)
* loss_gradient == d_y for the previous layers backward pass

- batched backward pass:
```cpp
vector<vector<double>> loss_gradient = layer_instance.backward(X, d_y);
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

[<- back to  Documentation](../README.md)