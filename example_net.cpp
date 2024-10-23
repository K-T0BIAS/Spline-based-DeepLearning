#include <iostream>
#include <algorithm>
#include <iomanip>
#include <numeric> // For std::accumulate
#include "layers.hpp"

class nn{
    private:
        
        double loss;
        double loss_grad;
        
    public:
    std::vector<layer> layers;
    
    nn(int num_layers,std::vector<unsigned int> in,std::vector<unsigned int> out,std::vector<unsigned int> detail,std::vector<double> max);
    std::vector<double> forward(std::vector<double> x,bool normalize);
    std::vector<double> backward(std::vector<double> x,std::vector<double> d_y,std::vector<double> y);
        
};

nn::nn(int num_layers,std::vector<unsigned int> in,std::vector<unsigned int> out,std::vector<unsigned int> detail,std::vector<double> max){
    loss=0.0;
    loss_grad=0.0;
    //create layer vector to hold futur layers
    std::vector<layer> new_layers;

    //initthe layers
    for (int i=0;i<num_layers;i++){
        new_layers.push_back(layer(in[i],out[i],detail[i],max[i]));
    }
    //assign laye4s
    layers=new_layers;//could throw error if so remove the std move call and directly assign new layer
}

std::vector<double> nn::forward(std::vector<double> x,bool normalize){
    //call forward for all layers
    for (int i=0; i<layers.size();i++){
        
        
        if(i!=layers.size()-1){
            x=layers[i].forward(x,normalize);
        }
        else{
            x=layers[i].forward(x,false);
        }
        
    }
    //return x <= prediction value
    return x;
}

std::vector<double> nn::backward(std::vector<double> x,std::vector<double> d_y,std::vector<double> y){
    //call backward for all oayers from last to first
    for (int i=layers.size()-1;i>=0;i--){
        x=(i>0) ? layers[i-1].last_output:x;
        d_y=layers[i].backward(x, d_y, y);
        
    }
    //return error gradient / loss gradient
    return d_y;
}
//standdart mse func
std::vector<double> mse_loss_gradient(const std::vector<double>& pred, const std::vector<double>& target) {
    std::vector<double> loss_grad(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        loss_grad[i] = 2 * (  target[i]- pred[i]);
    }
    return loss_grad;
}


int main() {
    //create network
    nn example_net = nn(1, {2}, {2}, {8}, { 1.0});
    //num of test samples
    int num_samples = 5;
    //init samples
    std::vector<std::vector<double>> Y(num_samples, std::vector<double>(2, 0.0));
    std::vector<std::vector<double>> X(num_samples, std::vector<double>(2, 0.0));
    
    // Sample data initialization
    for (int i = 0; i < num_samples; i++) {
        X[i][0] = static_cast<double>(i) / static_cast<double>(num_samples);  // Example input feature
        X[i][1] = static_cast<double>(i) / static_cast<double>(num_samples);  // Another feature
        Y[i][0] = X[i][0] * 20; // Example target value
        Y[i][1] = X[i][1] * 20; // Another target value
    }

    int num_epochs = 100; // Define number of epochs
    double learning_rate = 1; // Learning rate
    

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

        std::cout << "\n--- Epoch " << epoch + 1 << " ---" << std::endl;

        for (int sample = 0; sample < num_samples; sample++) {
            // Forward pass
            auto pred = example_net.forward(X[sample],true);

            // Print intermediate predictions
            std::cout << "Sample " << sample + 1 << " - Predictions: ";
            for (const auto& p : pred) {
                std::cout << std::fixed << std::setprecision(6) << p << " ";
            }
            std::cout << std::endl;

            // Compute the loss gradient
            auto loss_grad = mse_loss_gradient(pred, Y[sample]);
            total_loss += std::accumulate(loss_grad.begin(), loss_grad.end(), 0.0) / num_samples;

            // Print intermediate loss gradient
            std::cout << "Sample " << sample + 1 << " - Loss gradient: ";
            for (const auto& lg : loss_grad) {
                std::cout << std::fixed << std::setprecision(6) << lg << " ";
            }
            std::cout << std::endl;
            
            // Backward pass
            example_net.backward(X[sample], loss_grad, pred);

        }

        // Print the average loss for the epoch
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << std::fixed << std::setprecision(6) << total_loss << std::endl;
    }

    return 0;
}