## CPP CTensor Documentation

### include

first include the library header

```cpp 
#include "SplineNetLib/SplineNet.hpp"
```

### CTensor constructors

The CTensor class is usefull to perform tensor operations while automatically tracking the operations that a CTensor was involved with.
A CTensor stores the N dimensional data in a flat projected vector (std::vector<T>) alongside it's actual shape (std::vector<size_t>).
It will also store all arithmetic functions that it was used in or created from in a grad_fn vector (std::vector<std::unique_ptr<Function<T>>>). Important to note here is that a CTensor only gets a new grad_fn if it was the direct result of an operation (e.g. c = a + b , here only c gets the grad_fn entry).
grad_fns are classes that hold information about the parents of a CTensor (e.g. c = a + b, here c gets a new grad_fn that knows that a and b are the parents). They also have functions that determine the behaviour of the gradient propagation. 
Calling the backward function on one CTensor will automatically calculate the respective gradients of all other CTensors in the graph.

**Note** that the CTensor architecture was inspired by the pytorch tensor architecture. Read more here : [pytorch](https://github.com/pytorch/pytorch)

CTensors have multiple constructor options:

1: construct from nested vector

```cpp
std::vecor<std::vector<float>> data = {{1,2,3},{4,5,6}};

auto CTensor_instance = SplineNetLib::CTensor(data);
```

this creates a CTensor of shape {2,3}.
**Note** that new CTensors always have their requires gradient flag set to True.

2: construct from flat initializer list with initializer list of shape:

```cpp
auto CTensor_instance = SplineNetLib::CTensor({1.0,2.0,3.0,4.0,5.0,6.0}, {2,3});
```

this will result in the same CTensor as in the previous constructor

3: construct from flat vector and shape

```cpp
std::vector<size_t> shape = {2,3};
std::vector<float> data = {{1,2,3},{4,5,6}};

auto CTensor_instance = SplineNetLib::CTensor(data, shape);
```

4: construct from existing CTensor (shallow copy)

```cpp
auto first_CTensor = SplineNetLib::CTensor({1.0,2.0,3.0,4.0,5.0,6.0}, {2,3});

auto new_CTensor = SplineNetLib::CTensor(first_CTensor);
```

**Note** this creates a shallow copy any changes to each will affect the other

4.1: deep copy / clone

If a exact copy of a CTensor, that is independent, is needed do:

```cpp
auto first_CTensor = SplineNetLib::CTensor({1.0,2.0,3.0,4.0,5.0,6.0}, {2,3});

auto new_CTensor = first_CTensor.clone();
```

this will create a deep copy of "first_CTensor"

### CTensor getter functions

#### data()

this returns the inner data vector from the CTensor **Note** that this data vector is the flat representation of the CTensor.

example:

```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,3,4,5,6},{2,3});
auto data = CTensor_instance.data();
```

here data will be a vector<T> like {1,2,3,4,5,6}, where 'T' is the datatype of CTensor_instance.

#### shape()

this returns the shape of the CTensor

example:
```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,3,4,5,6},{2,3});
auto data = CTensor_instance.shape();
```

this returns a vector<size_t> = {2,3}.

### CTensor shape related functions

#### squeeze

squeeze will remove the indexed dimension from the shape. **Note** that the tensor size will remain the same and the size of the adjacent dimension will increase.

syntax:
```cpp
Ctensor.squeeze(size_t dim);
```

example:
```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,3},{1,3});
CTensor_instance.squeeze(0);
```

this will turn shape (1,3) into (3)

#### unsqueeze

unsqueeze will add a dimension of size 1 at the given indexed

syntax:
```cpp
Ctensor.unsqueeze(size_t dim);
```

example:
```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,3},{3});
CTensor_instance.unsqueeze(0);
```

this turns CTensor with shape (3) to CTensor with shape (1,3)

#### expand

expand can increase the size of the selected dimension by a factor n. The data at the seoected dimension will be copied and appended n times.

syntax:
```cpp
Ctensor.unsqueeze(size_t dim, int factor);
```

example:
```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,3},{1,3});
CTensor_instance.expand(0, 3);
```

the shape (1,3) becomes (3,3) and the data

((1,2,3)) becomes ↓

((1,2,3),  
 (1,2,3),  
 (1,2,3))

#### permute 

swaps around dimension sizes 

syntax:

syntax:
```cpp
Ctensor.permute(index_vector);
```
example:
```cpp
auto CTensor_instance = SplineNetLib::CTensor({1,2,1,2,1,2,1,2},{1,4,2});
std::vector<size_t> index_vector = {0,2,1};
```

the shape (1,4,2) will become (1,2,4). **Note** that this will not change the actual data vector as the permutation only affects the projection logic, meaning that when indexing a permutated CTensor the result will be different to before the permutation although the underlaying data is the same.

#### transpose

this transposes the CTensor meaning it swaps the inner most two dimensions (including the data in the flat vector)

syntax:

```cpp
Ctensor.transpose();
```

example: 

```cpp
auto CTensor_instance = SplineNetLib::CTensor({1.0,2.0,3.0,4.0,5.0,6.0}, {2,3});

CTensor_instance.transpose();
```

this will swap dim0 and dim1, so shape (2,3) becomes (3,2). The data vector [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] will change to [1.0, 4.0, 2.0, 5.0, 3.0, 6.0] to fit the new shape.



**more coming soon**

[<- back to Documentation](../README.md)