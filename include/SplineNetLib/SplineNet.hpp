#ifndef SPLINENET_HPP
#define SPLINENET_HPP

#include "layers.hpp"

namespace SplineNetLib {

//network class
class nn{
    
    public:
    //vector to store layers
    std::vector<layer> layers;
    //constructor to create network from scratch
    nn(int num_layers,std::vector<unsigned int> in,std::vector<unsigned int> out,std::vector<unsigned int> detail,std::vector<double> max);
    //forward pass (uses parameters for layer.forward)
    std::vector<double> forward(std::vector<double> x,bool normalize);
    //backward pass (uses parameters for layer.backward)
    std::vector<double> backward(std::vector<double> x,std::vector<double> d_y);
        
};

}//namespace

#endif