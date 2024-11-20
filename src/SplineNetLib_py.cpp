#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To handle STL types like std::string, std::vector
#include "SplineNetLib/SplineNet.hpp"    // Header for the library

namespace py = pybind11;

PYBIND11_MODULE(PySplineNetLib, m) {
    py::class_<SplineNetLib::spline>(m, "spline")
        .def(py::init<const std::vector < std::vector < double>>&, const std::vector < std::vector < double>>& >())  // Bind constructor
        .def("interpolation",&SplineNetLib::spline::interpolation,"None (None), interpolates the spline based on its points")
        .def("forward",&SplineNetLib::spline::forward,"double (double x), evaluates spline at x (if x in bounds)")
        .def("backward",&SplineNetLib::spline::backward,"double (double in,double d_y,double out), uses previous input, loss gradient and last output for gradient descent")
        .def("apply_grad",&SplineNetLib::spline::apply_grad,"None (double lr),apply grad from backward * lr")
        .def("get_points",&SplineNetLib::spline::get_points,"[[double]] (None),return spline points like [[x0,y0],...,[xn,yn]]")
        .def("get_params",&SplineNetLib::spline::get_params,"[[double]] (None),return spline parameters/coefficients like [[a0,b0,c0,d0],...,[an,bn,cn,dn]]");
    py::class_<SplineNetLib::layer>(m, "layer")
        .def(py::init<unsigned int, unsigned int, unsigned int, double>())
        .def(py::init<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<std::vector<std::vector<double>>>> >())
        .def("interpolate_splines",&SplineNetLib::layer::interpolate_splines,"None (None), calls interpolation on all splines in the layer")
        .def("forward",py::overload_cast<std::vector<double>, bool>(&SplineNetLib::layer::forward),"[double] ([double] x, bool normalize), forward call for single input sample")
        .def("forward",py::overload_cast<const std::vector<std::vector<double>> &, bool>(&SplineNetLib::layer::forward),"[[double]] (const [[double]] &x, bool normalize), forward call for batches")
        .def("backward",py::overload_cast<std::vector<double>,std::vector<double> , bool>(&SplineNetLib::layer::backward),"[double] ([double] x,[double]d_y,bool normalize), takes input x, loss gradient d_y and bool apply_grad,returns propageted loss (applies grad to all splines if True)")
        .def("backward",py::overload_cast<const std::vector<std::vector<double>> &,std::vector<std::vector<double>> >(&SplineNetLib::layer::backward),"backward but for batches (will always apply gradients)")
        .def("get_splines",&SplineNetLib::layer::get_splines,"[[SplineNetLib::spline]] (None), returns all splines in the layer");
}

/*to be checked
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To handle STL types like std::string, std::vector
#include "SplineNetLib/SplineNet.hpp"    // Header for the library

namespace py = pybind11;

PYBIND11_MODULE(mylibrary, m) {
    // Binding the spline class
    py::class_<SplineNetLib::spline>(m, "spline")
        .def(py::init<const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&>(),  // Constructor
             "Constructs the spline with control points and parameters")
        .def("interpolation", &SplineNetLib::spline::interpolation, 
             "None -> Interpolates the spline based on its points")
        .def("forward", &SplineNetLib::spline::forward, 
             "double (double x) -> Evaluates the spline at x (if x is in bounds)")
        .def("backward", &SplineNetLib::spline::backward,
             "double (double in, double d_y, double out) -> Uses previous input, loss gradient, and last output for gradient descent")
        .def("apply_grad", &SplineNetLib::spline::apply_grad,
             "None (double lr) -> Applies gradient from backward * learning rate (lr)")
        .def("get_points", &SplineNetLib::spline::get_points,
             "[[double]] -> Returns spline points like [[x0, y0], ..., [xn, yn]]")
        .def("get_params", &SplineNetLib::spline::get_params,
             "[[double]] -> Returns spline parameters/coefficients like [[a0, b0, c0, d0], ..., [an, bn, cn, dn]]");

    // Binding the layer class
    py::class_<SplineNetLib::layer>(m, "layer")
        .def(py::init<unsigned int, unsigned int, unsigned int, double>(),  // Constructor with size and learning rate
             "Constructs a layer with the specified number of splines and learning rate")
        .def(py::init<std::vector<std::vector<std::vector<std::vector<double>>>>, 
                      std::vector<std::vector<std::vector<std::vector<double>>>>>(),  // Constructor for nested vector input
             "Constructs a layer with nested vector inputs for spline initialization")
        .def("interpolate_splines", &SplineNetLib::layer::interpolate_splines,
             "None -> Calls interpolation on all splines in the layer")
        
        // Overloaded 'forward' methods
        .def("forward", py::overload_cast<std::vector<double>, bool>(&SplineNetLib::layer::forward), 
             "[double] (x, bool normalize) -> Forward call for single input sample, applies normalization if needed")
        .def("forward", py::overload_cast<const std::vector<std::vector<double>>&, bool>(&SplineNetLib::layer::forward), 
             "[[double]] (x, bool normalize) -> Forward call for batch inputs, applies normalization if needed")
        
        // Overloaded 'backward' methods
        .def("backward", py::overload_cast<std::vector<double>, std::vector<double>, bool>(&SplineNetLib::layer::backward),
             "[double] (x, d_y, bool normalize) -> Backward propagation for single input sample, applies grad if normalize is True")
        .def("backward", py::overload_cast<const std::vector<std::vector<double>>&, std::vector<std::vector<double>>>(&SplineNetLib::layer::backward),
             "[[double]] (x, d_y) -> Backward propagation for batch inputs, always applies gradients")
        
        .def("get_splines", &SplineNetLib::layer::get_splines,
             "[[SplineNetLib::spline]] -> Returns all splines in the layer");
}
*/