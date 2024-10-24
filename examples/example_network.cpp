#include <algorithm>
#include <iomanip>
#include <numeric> // For std::accumulate
#include "../include/SplineNetLib/SplineNet.hpp"

using namespace SplineNetLib;

//simple mse function to calculate loss gradient
std::vector<double> mse_loss_gradient(const std::vector<double>& pred, const std::vector<double>& target) {
    std::vector<double> loss_grad(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        loss_grad[i] = 2 * (  target[i]- pred[i]);
    }
    return loss_grad;
}

int main() {
    //create network (2 layers layer 1 :2 inputs 3 outputs , layer 2 :3 inputs 2 outputs,
    //detail for both = 8 (so 10 points per spline), points for both from 0 to 1,)
    nn example_net = nn(2, {2,3}, {3,2}, {8,8}, {1.0,1.0});
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

    int num_epochs = 20; //number of epochs

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;
        //current epoch
        std::cout << "\n--- Epoch " << epoch + 1 << " ---" << std::endl;
        //itter over "dataset"
        for (int sample = 0; sample < num_samples; sample++) {
            // Forward pass
            auto pred = example_net.forward(X[sample],true);

            // Print intermediate predictions
            std::cout << "Sample " << sample + 1 << " - Predictions: ";
            for (const auto& p : pred) {
                std::cout << std::fixed << std::setprecision(6) << p << " ";
            }
            std::cout << "\n";

            // Compute the loss gradient
            auto loss_grad = mse_loss_gradient(pred, Y[sample]);
            //accumaulate the loss of this epoch
            total_loss += std::accumulate(loss_grad.begin(), loss_grad.end(), 0.0) / num_samples;

            // Print intermediate loss gradient
            std::cout << "Sample " << sample + 1 << " - Loss gradient: ";
            for (const auto& lg : loss_grad) {
                std::cout << std::fixed << std::setprecision(6) << lg << " ";
            }
            std::cout << "\n";
            
            // Backward pass with current X inputs
            example_net.backward(X[sample], loss_grad, pred);

        }

        // Print the average loss for the epoch
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << std::fixed << std::setprecision(6) << total_loss << "\n";
    }

    return 0;
}