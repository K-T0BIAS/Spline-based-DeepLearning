// spline.h
#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>

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



class spline {
private:
    std::vector<std::vector<double>> params; // n-1 x m
    std::vector<std::vector<double>> points; // n x 2
    

public:
    
    spline(const std::vector<std::vector<double>> points_list,const std::vector<std::vector<double>> params_list);
    //default constructor do not use exept to reserve memory
    spline(){};

    // Member function for interpolation (assuemes points and params are inittialized)
    void interpolation();
    
    double forward(double x);
    
    //takes used x value, next layers loss gradient,target and learning rate,, returns this layers loss gradient
    double backward(double x,double d_y,double y,double lr);
    
    std::vector<std::vector<double>> get_points(); 
    
    std::vector<std::vector<double>> get_params();
};

#endif // SPLINE_HPP