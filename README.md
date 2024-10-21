# About Spline-based-DeepLearning

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
