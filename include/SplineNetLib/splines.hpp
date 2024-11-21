// Copyright (c) <2024>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// 
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0


#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
/*
#include <thread>
#include <mutex>
*/
namespace SplineNetLib {

// Function for zero argument error message
inline void print_err() {
    std::cerr << "\n";
}
//Full definition here bc of template
// Function to print error messages (variadic templates)
template<typename T, typename... Args>
void print_err(T first, Args... args) {
    std::cerr << first;
    print_err(args...);
}

extern bool parallel;


class spline {
private:
    std::vector<std::vector<double>> params; // n-1 x m
    std::vector<std::vector<double>> points; // n x 2
    
    std::vector<double> grad; //gradient where indx i == segment of spline that uses grad[i]
    
    
    
    //std::vector<double> batch_outputs; // shape 1d : (batchsize,) cached ouptus from latest fwd pass for the gradient calculation in backward, index by batch

public:
    
    spline(const std::vector<std::vector<double>> points_list,const std::vector<std::vector<double>> params_list);
    //default constructor do not use exept to reserve memory
    spline(){};

    // Member function for interpolation (assuemes points and params are inittialized)
    void interpolation();
    
    double forward(double x);
    
    //takes used x value, next layers loss gradient,target, returns this layers loss gradient
    double backward(double x,double d_y,double y);
    
    void apply_grad(double lr);
    
    
    std::vector<std::vector<double>> get_points(); 
    
    std::vector<std::vector<double>> get_params();
};

}//namespace

#endif // SPLINE_HPP