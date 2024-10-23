#include "layers.hpp"

//Notes:
//to remove point update output after every backward call:
//comment smartSplines.cpp line 145-148 
//in future this will need a getterfunc to retrieve points

std::vector<double> mse_loss_gradient(const std::vector<double>& pred, const std::vector<double>& target) {
    std::vector<double> loss_grad(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        loss_grad[i] = 2 * (  target[i]- pred[i]);
    }
    return loss_grad;
}

void test_spline_nn() {
    // Setup
    int in_size = 1;
    int out_size = 1;
    std::vector<double> x = {0.5, 1.0,1.1, 1.5};  // Sample inputs
    std::vector<double> y = {1.0,2.0,2.5,3.0};  //Sample targets
    layer layer_instance = layer(in_size, out_size, 18,2.0);
    
    layer_instance.interpolate_splines();  // Initialize splines
    layer_instance.lr = 1;  // Learning rate (1 works surprisingly well)

    // Expected results (modify as per actual implementation details)
    std::vector<double> expected_grads = {-1.0, -2.0, -3.0};  // Example values, replace accordingly
    for (int j=0;j<10;j++){
        for (size_t i = 0; i < x.size(); ++i) {
            std::vector<double> input = {x[i]};
            std::vector<double> target ={y[i]};
            std::vector<double> pred = layer_instance.forward(input,false);
            std::vector<double> loss_grad = mse_loss_gradient(pred, target);
    
            std::cout << "\n loss grad: " << loss_grad[0] 
                      << " pred: " << pred[0] 
                      << " target: " << target[0] << "\n";

            // Check backward pass
            auto gradient = layer_instance.backward( input, loss_grad,pred);
            std::cout << "grad: " << gradient[0] << "\n";
    

        }
    }
    
}

int main() {
    test_spline_nn();
    return 0;
}