/*
new:
batch output cached
hpp l.37
cpp l.115-119

new: 
grad calcupation and apply now sepperate 
hpp l.42 & 44
cpp rm l.155-158
add l.162-170
todo : rm lr from spline.backward
---> also in layer cpp added apply call
*/
// spline.h
#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <thread>
#include <mutex>

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
    
    //takes used x value, next layers loss gradient,target and learning rate,, returns this layers loss gradient
    double backward(double x,double d_y,double y);
    
    void apply_grad(double lr);
    
    
    std::vector<std::vector<double>> get_points(); 
    
    std::vector<std::vector<double>> get_params();
};

}//namespace

#endif // SPLINE_HPP