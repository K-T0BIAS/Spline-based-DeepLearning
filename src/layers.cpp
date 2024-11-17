#include "../include/SplineNetLib/layers.hpp"
//added this line to test the tests
namespace SplineNetLib {

template <typename T>//new
layer::layer(unsigned int _in_size, unsigned int _out_size, unsigned int _detail,double max,std::shared_ptr<T> _activation) {
    
    activation=_activation;
    in_size=_in_size;
    out_size=_out_size;
    detail=_detail;
    // Prepare l_splines vector
    l_splines.resize(_in_size); // Resize the outer vector
    for (size_t i = 0; i < _in_size; i++) {
        l_splines[i].resize(_out_size); // Resize each inner vector
    }
    
    //create zeroed points vector
    std::vector < std::vector < double>>points(_detail + 2, std::vector < double > (2));
    //counter for x coordinate
    double counter = 0.0;
    //increment x value based on number of points so that all x are spaced evenly
    for (int i = 1; i < _detail+1; i++) {
        counter += max/(_detail+1.0);//increment count
        points[i][0] = counter;//assign count to x var
    }
    //maje sure that last point ist exactly max value
    points[points.size()-1][0]=max;
    
    // Create spline with default points and params
    for (size_t i = 0; i < _in_size; i++) {
        for (size_t j = 0; j < _out_size; j++) {
            // Directly assign the splines
            l_splines[i][j] = spline(
                points, // points
                std::vector < std::vector < double>>(_detail + 1, std::vector < double > (4)) // params
                );
        }
    }
}

template <typename T>//new
layer::layer(std::vector < std::vector < std::vector < std::vector < double>>>> points_list,
             std::vector < std::vector < std::vector < std::vector < double>>>> params_list,
             std::shared_ptr<T> _activation){//new
    
    activation=_activation;
    in_size = points_list.size();
    out_size = points_list[0].size();
    detail = points_list[0][0].size() - 2; // must be > 0

    // Safety check to ensure points and params have the same first 3 dimensions
    if (params_list.size() != in_size || params_list[0].size() != out_size || params_list[0][0].size() != detail + 1) {
        throw std::invalid_argument("points_list and params_list must have matching dimensions.");
    }

    // create l_splines vector
    l_splines.resize(in_size); // Resize the outer vector
    for (size_t i = 0; i < in_size; i++) {
        l_splines[i].resize(out_size); // Resize each inner vector
    }

    // Create spline with points and params list
    for (size_t i = 0; i < in_size; i++) {
        for (size_t j = 0; j < out_size; j++) {
            // Directly assign the unique_ptr returned by spline::create to avoid copy error
            l_splines[i][j] = spline(points_list[i][j], params_list[i][j]);
        }
    }
    
}

void layer::interpolate_splines() {
    for (size_t i = 0; i < l_splines.size(); ++i) {
        for (size_t j = 0; j < l_splines[i].size(); ++j) {
            l_splines[i][j].interpolation(); //imterpolate all splines in layer
        }
    }
}


std::vector < double > layer::forward(std::vector < double> x,bool normalize) {
    
    //std::cout<<"layer fwd call\n";
    // Initialize output with zeros
    std::vector < double > output(out_size, 0.0);
/*
    // Debug: Print the input vector
    std::cout << "Input vector x: ";
    for (double val : x) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
*/
    for (size_t i = 0; i < in_size; i++) {
        for (size_t j = 0; j < out_size; j++) {
            // Get the spline value for input x[i]
            double spline_output = l_splines[i][j].forward(x[i]);

            // sum the output from this spline into the output vector
            output[j] += spline_output;
            /*
            // Debug: Print the output from the current spline
            std::cout << "Spline[" << i << "][" << j << "] output for x[" << i << "]: " << spline_output << std::endl;
*/
        }
    }
    /*
    // Debug: Print the final output vector
    std::cout << "Final output vector: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
*/
    if (normalize){
        double max=output[0];
        for (double x:output){
            max=(max<x) ? x:max;
        }
        if (max!=0){
            for (int i=0;i<output.size();i++){
                output[i]/=max;
            }
        }
    }
    last_output=output;
    //new
    //check if activation exists and apply activation to all outputs
    if (activation != nullptr) {
        for (size_t i=0; i < out_size; i++) {
            output[i] = activation->forward(output[i])
        }
    }
    
    return output;
}

std::vector < double > layer::backward(std::vector < double > x, std::vector < double > d_y, std::vector < double > y) {

    std::vector < double > out(in_size, 0.0);
    std::vector < std::vector < double>> spline_outputs(out_size, std::vector < double > (in_size));
    std::vector < double > total_outputs(out_size, 0.0);

    // Compute spline outputs and sum them up like in forward (cant use forward bc i need both outputs)
    for (size_t j = 0; j < out_size; j++) {
        for (size_t i = 0; i < in_size; i++) {
            spline_outputs[j][i] = l_splines[i][j].forward(x[i]); 
            total_outputs[j] += spline_outputs[j][i]; // Sum up outputs from splines
        }
    }

    // Now calculate the gradients based on individual spline contribution
    for (size_t i = 0; i < in_size; i++) {
        for (size_t j = 0; j < out_size; j++) {
            double spline_output = spline_outputs[j][i];
            double total_output = total_outputs[j];

            // compute the contribution of each spline
            double contribution_ratio = 1;//default to 1 for now (better to 0 when points[i][1] -> y is initialized to !=0)

            if (total_output != 0.0) {
                contribution_ratio = spline_output / total_output;
            }

            // The gradient for this spline is the total gradient scaled by its contribution to the output sum
            double adjusted_gradient = d_y[j] * contribution_ratio;
            
            //new
            // calculate the gradient of the activation and adjust spline gradient based on it
            if (activation != nullptr) {
                adjusted_gradient = activation->backward(spline_output, adjusted_gradient);
            }

            // calculate gradients for the splines and sum them for the previous layer->backward pass
            out[i] += l_splines[i][j].backward(x[i],adjusted_gradient, spline_output,   lr);//ggf swap s out and adj grad
        }
    }

    return out;
}

}//namespace