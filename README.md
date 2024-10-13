About the project:
  Goals:
    -mimic linear networks with spline based layers
    -visualize forward passes using splines
    -achieve high precision similar to conventional neural networks
    -compatibility with other networks such as convolutional nn

  implemented:
    -spline class with functional forward , backward and interpolation function
    -layer class to manage splines as substitution for weights & bias with
     functional create 'new' or 'from parameters' initializer/factory method 
     aswell as forward and backwards methods to calculate layer outputs and gradient

  how to use:
    Step 1:
    case 1: creating new layer:
      1. as unsigned int:
         set input size (numer of inputs for this layer),
         output size (number of expected outputs for this layer)
         detail (number of control points for this spline)(splines start with 2 default points)
         (more detail means more precision but also more params)
         [num parameters = input size * output size * (detail + 1) * 4]
      2. create layer as std::unique_ptr<layer> layer_instance:
         with: layer::create(input size,output size, detail)
         Note: likely will change this to shared ptr in the future 
    case 2: creating layer with known points and parameters:
      1. create layer as std::unique_ptr<layer> layer_instance:
         with: layer::create(points,parameters)
         Note: likely will change this to shared ptr. in the future 
    Step 2:
      call: layer_instance->interpolate_splines()
        (this prepares the layer for the first forward pass)
      also set: layer_instance->lr (learning rate)(first tests have shown 1 to be quite effective)
    Step 3:
      set: (std::vector<double>)prediction = layer_instance->forward(input)(input must be vector)
    Step 4:
      calculate loss gradient with loss function of choice (MSE works well)
    Step 5:
      set: (std::vector<double> loss_gradient =layer_instance->backward(input,loss_grad,pred)
        (loss grad can be passed to the previous layer for backpropagation)
        (if layer is last layer input the loss gradient from the loss function instead)
    Step 6:
      repeat steps 2 to 5 for however many epochs are needed
    Step 7:
      no graphics implemented for the splines yet but to visualize the splines input the points of 
      any spline into a online simulator (must be set to cubic splines as that is what the program uses)

Note: 
  currently the main function tests for a one layer one input output pair instance over 10 epochs (with MSE loss)
   
      
      
