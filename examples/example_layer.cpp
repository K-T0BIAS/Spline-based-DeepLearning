#include "../include/SplineNetLib/layers.hpp"

using namespace SplineNetLib;

//mse loss func tondemonstrate training of singular layer
std::vector<double> mse_loss_gradient(const std::vector<double>& pred, const std::vector<double>& target) {
    std::vector<double> loss_grad(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        loss_grad[i] = 2 * (  target[i]- pred[i]);
    }
    return loss_grad;
}


int main (){
    
    //initialisation parameters for constructor 1
    int in_size = 2;
    int out_size = 1;
    int detail = 8;
    double max =1.0;
    
    std::vector<std::vector<double>> x = {{0.1,0.3},{0.3,0.2},{0.5,0.9}};  // Sample inputs
    std::vector<std::vector<double>> y = {{7},{7},{23}};  //Sample targets (x[i][0] + 2 * x[i][1])
    //create new layer from constructor A
    layer layer_A = layer(in_size,out_size,detail,max);
    //init layer-spline parameters
    layer_A.interpolate_splines();//only needed for layers created from constructor 1 but still good practice for constructor 2

    //create small training loop for 5 epochs
    for (int i = 0; i < 20; i++ ){
        std::cout<<"\nepoch : "<< i+1 <<"\n";
        //itter over "dataset"
        for (size_t j = 0; j < x.size(); j++){
            auto X = x[j];//input
            auto Y = y[j];//target
            //prediction || forward pass
            auto pred = layer_A.forward(X,false);//set output normalization to false otherwise outputs will be between 0. and 1.
            //claculate loss gradient for backward pass
            auto loss_grad = mse_loss_gradient(pred, Y);
            //perform backward pass
            auto layer_loss_grad = layer_A.backward(X,loss_grad);
            
            //print input, pred, target and loss grad 
            std::cout<< "input : "<<X[0]<<","<<X[1]<<" | prediction : "<<pred[0]<<" | target : "<<Y[0]<<" | loss gradient : "<<loss_grad[0]<<"\n";
        }
    }
    
    //example to use constructor 2
    //this layer takes 2 inputs and gives 3 outputs 
    std::vector<std::vector<std::vector<std::vector<double>>>> points_B = {{{{0.0,1.0},{0.5,2.0},{1.0,3.0}},
                                                                            {{0.0,0.1},{0.5,0.2},{1.0,0.4}},
                                                                            {{0.0,0.5},{0.5,1.0},{1.0,2.0}}},
                                                                          {{{0.0,1.0},{0.5,2.0},{1.0,4.0}},
                                                                            {{0.0,0.1},{0.5,0.2},{1.0,0.3}},
                                                                            {{0.0,2.0},{0.5,2.5},{1.0,2.5}}}
                                                                         };//2x3x3x2 = in size x out size x 2+detail x 2
                                                                         
    //parameters initialized to 0 for simplicity could be already a list of interpolation values
    std::vector<std::vector<std::vector<std::vector<double>>>> parameters_B (2,std::vector<std::vector<std::vector<double>>>(
                                                                            3,std::vector<std::vector<double>>(
                                                                            2,std::vector<double>(
                                                                            4,0.0))));//2x3x2x4 = in size x out size x 1+detail x 4
                                                                            
    //create layer
    layer layer_B = layer(points_B,parameters_B);
    //init parameters
    layer_B.interpolate_splines();
    //evaluate at x
    std::vector<double> x_B = {1.0,1.0};
    auto pred_B =layer_B.forward(x_B,false);
    
    std::cout<<"\n\nlayer B evaluation : "<<pred_B[0]<<","<<pred_B[1]<<","<<pred_B[2];
    
    return 0;
}