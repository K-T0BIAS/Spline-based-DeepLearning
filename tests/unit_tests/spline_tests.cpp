#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "../include/SplineNetLib/layers.hpp"

using namespace SplineNetLib ;

//test initialization
TEST_CASE("spline initialization using constructor method functions as expected") {
    std::vector<std::vector<double>> correct_points = {{0.0,0.0},{0.2,1.0},{0.4,2.5},{0.6,2.0},{0.8,2.0},{1.0,0.5}};
    std::vector<std::vector<double>> incorrect_dim_points = {{0.0,1.0},{0.5},{1.0,1.0}};
    std::vector<std::vector<double>> incorrect_num_points = {{0.0,0.0}};
    std::vector<std::vector<double>> correct_parameters (5, std::vector<double>(4,0.0));
    std::vector<std::vector<double>> incorrect_dim_parameters = {{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0,0.0}};
    std::vector<std::vector<double>> incorrect_num_parameters = {{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0}}
    
    REQUIRE_NOTHROW(spline(correct_points,correct_parameters));
    REQUIRE_THROW_AS(spline(incorrect_dim_points,correct_parameters),std::runtime_error);
    REQUIRE_THROW_AS(spline(incorrect_num_points,correct_parameters),std::runtime_error);
    REQUIRE_THROW_AS(spline(correct_points,incorrect_dim_parameters),std::runtime_error);
    REQUIRE_THROW_AS(spline(correct_points,incorrect_num_parameters),std::runtime_error);
    
    
    
}

TEST_CASE("spline interpolation functions as expected"){
    //these are the reference points and params that the spline shouod interpolate to
    std::vector<std::vector<double>> correct_points = {{0.0,0.0},{0.2,1.0},{0.4,2.5},{0.6,2.0},{0.8,2.0},{1.0,0.5}};
    std::vector<std::vector<double>> correct_parameters (5, std::vector<double>(4,0.0));
    
    spline Test_spline = spline(correct_points,correct_parameters);
    Test_spline.interpolation();
    std::vector<std::vector<double>> spline_points = Test_spline.get_points();
    std::vector<std::vector<double>> spline_parameters = Test_spline.get_params();
    
    //check if spline points and correct_points. if print both
    if (spline_points != correct_points){
        std::cout<<"spline points after interpolation do not match expected points: \n";
        std::cout<<"correct points = (";
        for (size_t i = 0; i < correct_points.size(); i++){
            std::cout<<"(";
            for size_t j = 0: j < correct_points[0].size(); i++){
                std::cout<<correct_points[i][j];
                (j < correct_points[0].size()-1) ? std::cout<<", " : std::cout<<")"
            }
            (i < correct_points.size()-1)? std::cout<<"), ": std::cout<<")\n";
        }
        
        std::cout<<"spline points = (";
        for (size_t i = 0; i < spline_points.size(); i++){
            std::cout<<"(";
            for size_t j = 0: j < spline_points[0].size(); i++){
                std::cout<<spline_points[i][j];
                (j < spline_points[0].size()-1) ? std::cout<<", " : std::cout<<")"
            }
            (i < spline_points.size()-1)? std::cout<<"), ": std::cout<<")\n";
        }
    }
    REQUIRE(spline_points == correct_points);
    
    if (spline_parameters != correct_parameters){
        std::cout<<"spline parameters after interpolation do not match expected parameters: \n";
        std::cout<<"correct parameters = (";
        for (size_t i = 0; i < correct_parameters.size(); i++){
            std::cout<<"(";
            for size_t j = 0: j < correct_parameters[0].size(); i++){
                std::cout<<correct_parameters[i][j];
                (j < correct_parameters[0].size()-1) ? std::cout<<", " : std::cout<<")"
            }
            (i < correct_parameters.size()-1)? std::cout<<"), ": std::cout<<")\n";
        }
        
        std::cout<<"spline parameters = (";
        for (size_t i = 0; i < spline_parameters.size(); i++){
            std::cout<<"(";
            for size_t j = 0: j < spline_parameters[0].size(); i++){
                std::cout<<spline_parameters[i][j];
                (j < spline_parameters[0].size()-1) ? std::cout<<", " : std::cout<<")"
            }
            (i < spline_parameters.size()-1)? std::cout<<"), ": std::cout<<")\n";
        }
    }
    REQUIRE(spline_parameters == correct_parameters);
}

