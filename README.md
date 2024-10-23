# About Spline-based-DeepLearning

# attention after last merge initializing methods no longer use factory methods instead they now use constructors the readme will be updated accordingly in the near future

## goals:

1. create visual representations for neural networks by replacing commonly used fully connected layers with spline based layers.
2. achieve similar or better precision to common deep learning approaches whilst keeping the structure as light-wheight and fast as possible.
3. allow easy adaptability to existing architectures like convolutional and recurrent networks.

## Implementation:

### Splines:
The splines are the main computation unit of a layer. They allow for an easily adjustable and visualizable alternative to wheight matricies.
To create a spline call:
```cpp
std::unique_ptr<spline> spline_instance = spline::create(points,parameters);
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
Spline_instance->interpolate();
```
this, although not always nessecery  will adjust the parameters with respect to the points.

To evaluate the spline at point x do:
```cpp
double y = Spline_instance->forward(x)
```
**Note** that x must be between 0 and the largest x value in the splines points list. Trying to access x values outside the spline will result in an error.

To perform a backward pass call:
```cpp
double loss_grad = spline->backward(...
```


## License

This project is licensed under the Mozilla Public License 2.0. 

Copyright (c) 2024 Tobias Karusseit. See the [LICENSE](./LICENSE) file for details.
