#include "../include/splines.hpp"

int main (){
    //create points for spline (must be vector < vector < double >>)
    std::vector<std::vector<double>> points = {{0.0,0.0},{0.2,1.0},{0.4,2.0},{0.6,3.0},{0.8,4.0},{1.0,5.0}};//6x2 vector
    //create parameters (can be all 0.) (parameters.size() must be = points.size()-1)(parameters[i].size() must be = 4)
    std::vector<std::vector<double>> parameters (5, std::vector<double>(4,0.0)); //5x4 vector 
    //create the spline using constructor (NOTE using default constructor spline(); is not recomended as the spline
    //is unusable since the points are not set properly and cannot be changed manually later.
    //This spline could only function as a placeholder in memory e.g in src/layers.cpp constructor)
    spline spline_instance = spline(points,parameters);
    //use interpolation method to initialize the parameters
    spline_instance.interpolation();
    //evaluate spline a point x (points.[0][0] <= x <= points[points.size()-1][0] or x between min and max x coordintes of spline)
    double eval = spline_instance.forward(0.5);// eval at x = 0.5
    //to get points and parameters use getter methods: .get_points || .get_params
    std::vector<std::vector<double>> spline_points = spline_instance.get_points();
    std::vector<std::vector<double>> spline_parameters = spline_instance.get_params();
    
    //print points params and eval
    std::cout<<"points\n";
    for (const auto &point : spline_points){
        std::cout<< "\nx = " << point[0] << "\n";
        std::cout<< "y = " << point[1] << "\n";
    }
    std::cout<<"\nparameters\n";
    for (const auto &params : spline_parameters){
        std::cout<< "\na = " << params[0] << "\n";
        std::cout<< "b = " << params[1] << "\n";
        std::cout<< "c = " << params[2] << "\n";
        std::cout<< "d = " << params[3] << "\n";
    }
    std::cout<<"\nevaluation at 0.5 = " << eval << "\n";
    
    return 0;
}