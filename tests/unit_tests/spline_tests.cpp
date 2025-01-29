#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/SplineNetLib/layers.hpp"

using namespace SplineNetLib;

// Test initialization
TEST_CASE("spline initialization using constructor method functions as expected") {
    std::vector<std::vector<double>> correct_points = {{0.0, 0.0}, {0.2, 1.0}, {0.4, 2.5}, {0.6, 2.0}, {0.8, 2.0}, {1.0, 0.5}};
    std::vector<std::vector<double>> incorrect_dim_points = {{0.0, 1.0}, {0.5}, {1.0, 1.0}};
    std::vector<std::vector<double>> incorrect_num_points = {{0.0, 0.0}};
    std::vector<std::vector<double>> correct_parameters(5, std::vector<double>(4, 0.0));
    std::vector<std::vector<double>> incorrect_dim_parameters = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    std::vector<std::vector<double>> incorrect_num_parameters = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    REQUIRE_NOTHROW(spline(correct_points, correct_parameters));
    REQUIRE_THROWS_AS(spline(incorrect_dim_points, correct_parameters), std::runtime_error);
    REQUIRE_THROWS_AS(spline(incorrect_num_points, correct_parameters), std::runtime_error);
    REQUIRE_THROWS_AS(spline(correct_points, incorrect_dim_parameters), std::runtime_error);
    REQUIRE_THROWS_AS(spline(correct_points, incorrect_num_parameters), std::runtime_error);
}

TEST_CASE("spline interpolation functions as expected") {
    std::vector<std::vector<double>> correct_points = {{0.0, 0.0}, {0.2, 0.0}, {0.4, 0.0}, {0.6, 0.0}, {0.8, 0.0}, {1.0, 0.0}};
    std::vector<std::vector<double>> correct_parameters(5, std::vector<double>(4, 0.0));

    spline Test_spline(correct_points, correct_parameters);
    Test_spline.interpolation();
    std::vector<std::vector<double>> spline_points = Test_spline.get_points();
    std::vector<std::vector<double>> spline_parameters = Test_spline.get_params();

    // Check if spline points match expected points
    if (spline_points != correct_points) {
        std::cout << "Spline points after interpolation do not match expected points:\n";
        std::cout << "Correct points: ";
        for (const auto& point : correct_points) {
            std::cout << "(";
            for (size_t j = 0; j < point.size(); ++j) {
                std::cout << point[j] << (j < point.size() - 1 ? ", " : "");
            }
            std::cout << ") ";
        }
        std::cout << "\nSpline points: ";
        for (const auto& point : spline_points) {
            std::cout << "(";
            for (size_t j = 0; j < point.size(); ++j) {
                std::cout << point[j] << (j < point.size() - 1 ? ", " : "");
            }
            std::cout << ") ";
        }
        std::cout << "\n";
    }
    REQUIRE(spline_points == correct_points);

    // Check if spline parameters match expected parameters
    if (spline_parameters != correct_parameters) {
        std::cout << "Spline parameters after interpolation do not match expected parameters:\n";
        std::cout << "Correct parameters: ";
        for (const auto& param : correct_parameters) {
            std::cout << "(";
            for (size_t j = 0; j < param.size(); ++j) {
                std::cout << param[j] << (j < param.size() - 1 ? ", " : "");
            }
            std::cout << ") ";
        }
        std::cout << "\nSpline parameters: ";
        for (const auto& param : spline_parameters) {
            std::cout << "(";
            for (size_t j = 0; j < param.size(); ++j) {
                std::cout << param[j] << (j < param.size() - 1 ? ", " : "");
            }
            std::cout << ") ";
        }
        std::cout << "\n";
    }
    REQUIRE(spline_parameters == correct_parameters);
}

