
## python Spline Implementation/documentation

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

or alternatively do:

```python
spline_instance = PySplineNetLib.spline([[0.0,0.0],[0.5,0.25],[1.0,1.0]],[[0.0]*4]*2)
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

y : float = actual target 

d_y : float = gradient of loss with (x,target) with respect to spline (x,y) (=> loss.backward() or d_y of next layer)

**Note :**

The gradient of this function call is internally stored in the spline.

- adjust spline based on gradient

to apply the gradient from .backward and adjust the spline call:
```python
spline_instance.apply_grad(lr)
```

lr : float = learning rate (controls how strong the gradient affects the splines points)

## layer

layers combine multiple splines to map an input vector of size m to an output vector of size n by evaluating splines at the input values and combining these outputs into the output. To achieve this the layer uses an m x n spline matrix where for every input<sub>i</sub> there exist n splines. 

mathematically the output $y$ is defined like this:

$$
y_j = \sum_{i=1}^{m} S_{i,j}(x_i), \quad \forall j \in \{1, \dots, n\}
$$

for example given input size 3 and output size 2, output<sub>1</sub> is the sum of splines<sub>i,1</sub> with i from 0 to 3 (input size)

To create a new layer do:

```python
layer_instance = PySplineNetLib.layer(input_size, output_size, detail, max)
```

where:

input_size : int = the size of the input vector
output_size : int = the expected size of the output vector
detail : int = the number of controlpoints for ecah spline (NOTE that the spline has detail + 2 points so to get 10 points detail shouod be 8)
max : float = the maximum value that any spline in the layer can evaluate (recomended 1.0 combined with activations that map input and output to range(0,1))

alternatively you can create a spline with start values for points and parameters like this:

```python
spline_instance = PySplineNetLib(points, parameters)
```

with:
points : list = nested list of points like : (input_size, output_size, detail +2, 2 = x,y) 
parameters : list = nested list of points like : (input_size, output_size, detail +1, 4)

to fully init the layer call:

```python
layer_instance.interpolate_splines()
```

### forward pass

```python
pred = layer_instance.forward(X)
```

where

X : list = single input vector or batched input vector
pred : list = prediction vector (also with batch dimension if the input was batched)

### backward pass

```python
d_y = layer_instance.backward(X, d_y)
```

where:

X is the last inputvthis layer recieved
d_y is the propagated gradient of the previous layer

Note that backward will apply the gradient to all splines in the layer automatically

[<- back to Documentation](../README.md)