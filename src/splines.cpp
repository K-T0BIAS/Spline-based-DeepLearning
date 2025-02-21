// Copyright (c) <2024>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0

#include "../include/SplineNetLib/splines.hpp"

namespace SplineNetLib {
    
bool parallel = false;


spline::spline(const std::vector < std::vector < double>> points_list, const std::vector < std::vector < double>> params_list){
    if (points_list.size() < 2) {
        throw std::runtime_error("to few points in points_list. (Minimum num points == 2)");
    }
    if(params_list.size() != points_list.size()-1) {
        throw std::runtime_error("invalid num of parameter lists. (params_list.size()==points_list.size()-1)");
    }
    for (size_t i = 0; i < points_list.size(); i++) {
        if (points_list[i].size() != 2) {
            print_err("inconsistent size in points_list dim 1 at index ", i, " size of dim1 must be 2");
            throw std::runtime_error("invalid points_list dimensions");
        }
    }
    for (size_t i = 0; i < params_list.size()-1; i++) {
        if (params_list[i].size() != 4) {
            print_err("inconsistent size of parameters at params_list[", i, "] (mustbbe ==4)");
            throw std::runtime_error("invalid params_list size");
        }
    }
    params = params_list;
    points = points_list;
    
    grad = std::vector<double> (params_list.size(),0.0);//vec of length of num of sub segments in spline
    //std::cout<<"params_list size "<<params.size()<<"\n";
}

void spline::interpolation() {
    //std::cout<<"interpolation call\n";


    //append temporay param list to store edge values in forward and backward
    params.push_back(std::vector < double > (4));

    int n = points.size() - 1; // Number of intervals
    if (n < 1) {
        throw std::runtime_error("Not enough points for interpolation.");
    }

    std::vector < double > h(n),
    alpha(n),
    l(n + 1),
    mu(n + 1),
    z(n + 1);
    // Compute h
    for (int i = 0; i < n; ++i) {
        h[i] = points[i + 1][0] - points[i][0];
    }

    // Compute alpha
    for (int i = 1; i < n; ++i) {
        alpha[i] = (3.0 / h[i]) * (points[i + 1][1] - points[i][1]) -
        (3.0 / h[i - 1]) * (points[i][1] - points[i - 1][1]);
    }

    // Initialize l, mu, and z
    l[0] = 1.0;
    mu[0] = z[0] = 0.0;

    // Forward sweep
    for (int i = 1; i < n; ++i) {
        l[i] = 2.0 * (points[i + 1][0] - points[i - 1][0]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    l[n] = 1.0;
    z[n] = params[n][2] = 0.0; // Assuming natural spline conditions

    // Back substitution
    for (int j = n-1; j >= 0; --j) {
        params[j][2] = z[j] - mu[j] * params[j+1][2];
        params[j][1] = (points[j+1][1] - points[j][1]) / h[j] - h[j] * (params[j+1][2] + 2.0 * params[j][2]) / 3.0;
        params[j][3] = (params[j+1][2] - params[j][2]) / (3.0 * h[j]);
        params[j][0] = points[j][1];
    }

    params.pop_back(); //remove temp placeholders
/*debug
    //to print interpolation result
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = 0; j < params[i].size(); ++j) {
            std::cout << params[i][j] << " "; // Print each value in the parameter vector
        }
        std::cout << "\n"; // New line after each parameter set
    }

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points[i].size(); ++j) {
            std::cout << points[i][j] << " "; // Print each value in the parameter vector
        }
        std::cout << "\n"; // New line after each parameter set
    }
*/
}

double spline::forward(double x) {
    //std::cout<<"spline fwd call\n";
    if (points.empty() || params.empty()) {
        throw std::runtime_error("No points or parameters defined for spline.");
    }
    
    // Find the interval that x belongs to
    size_t i;
    for (i = 1; i < points.size(); i++) {
        if (x <= points[i][0]) {
            // Use the previous spline segment's params
            x = x - points[i - 1][0]; // Adjust x relative to the spline
            // Perform cubic polynomial interpolation using the parameters
            return params[i - 1][0] + params[i - 1][1] * x + params[i - 1][2] * (x * x) + params[i - 1][3] * (x * x * x);
            //for output caching 
            /*
            double out = params[i - 1][0] + params[i - 1][1] * x + params[i - 1][2] * (x * x) + params[i - 1][3] * (x * x * x);
            batch_outputs.push_back(out);
            return out;
            */
            // Found the correct interval
        }
    }
    
    // x does not exist in the control points
    print_err("x not in range of spline bounds. bounds : [", points[0][0], ",", points[points.size() - 1][0], "]");
    throw std::runtime_error("x out of bounds");

}
//x =input from forward,y_d output from prev layer or error func,y= expected targed value,lr =learning rate
double spline::backward(double x, double d_y, double y) {
    //std::cout<<"backward in spline\n";
    //check for empty points and parameters
    if (points.empty() || params.empty()) {
        throw std::runtime_error("No points or parameters defined for spline.");
    }
    
    //find segment of x
    size_t i;
    for (i = 1; i < points.size(); i++) {

        if (x <= points[i][0]) {
            break; //exit if segment index is found
        }
    }
/*debug
    std::cout<<"\nin spline backwards x="<<x<<" founf point indx:"<<i<<"\n";
*/
    double d_E = (forward(x)-y)+d_y; //respective error of current layer + accumulated grad
/*debug
    std::cout<<"dy: "<<d_y<<"D_E in spline="<<d_E<<"\n";
*/
    grad[i] += d_E;
    /*
    points[i][1] = points[i][1]+lr*d_E; //Adjust y_i based on error grad
    //update spline
    interpolation(); //done still experimental
    */
    return d_E; //return error grad for backwards pass into next layer
}

void spline::apply_grad(double lr) {
    for (size_t i = 0; i < grad.size(); i++ ) {
        if (grad[i] != 0.0) {
            points[i][1] = points[i][1]-lr*grad[i]; //Adjust y_i based on error grad
            grad[i] = 0.0; //reset grad for next bwd 
        }
    }
    this->interpolation();
}

std::vector<std::vector<double>> spline::get_points(){
    return points;
}

std::vector<std::vector<double>> spline::get_params(){
    return params;
}

}//namespace

//adding this line to force run the new tests without making meaningfull changes to the code (ummay delete this later)