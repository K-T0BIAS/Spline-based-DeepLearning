#include "../include/SplineNetLib/SplineNet.hpp"

namespace SplineNetLib {

nn::nn(int num_layers,std::vector<unsigned int> in,std::vector<unsigned int> out,std::vector<unsigned int> detail,std::vector<double> max){
    
    //create layer vector to hold future layers
    std::vector<layer> new_layers;

    //init the layers
    for (int i=0;i<num_layers;i++){
        new_layers.push_back(layer(in[i],out[i],detail[i],max[i]));//layer constructor 1 without loading old parameters
    }
    //assign layers
    layers=new_layers;
}

std::vector<double> nn::forward(std::vector<double> x,bool normalize){
    //call forward for all layers
    for (int i=0; i<layers.size();i++){
        
        //normalize for all layers exept last one
        if(i!=layers.size()-1){
            x=layers[i].forward(x,normalize);
        }
        //final layer forward pass no normalization
        else{
            x=layers[i].forward(x,false);
        }
        
    }
    //x = prediction value
    return x;
}

std::vector<double> nn::backward(std::vector<double> x,std::vector<double> d_y){
    //call backward for all oayers from last to first
    for (int i=layers.size()-1;i>=0;i--){
        //if layers[i] is not the first layer use the previou layers (layer[i-1]) prediction as x since backward needs the input from the forward pass
        x=(i>0) ? layers[i-1].last_output : x;
        d_y=layers[i].backward(x, d_y);
        
    }
    //return error gradient || loss gradient
    return d_y;
}
    
}