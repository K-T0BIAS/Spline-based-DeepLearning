// Copyright (c) <2024>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// 
// This file also includes contributions from the pybind11 library, which is licensed 
// under the MIT License.
//
// SPDX-License-Identifier: MIT
//
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0
// - MIT License: https://opensource.org/licenses/MIT


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // To handle STL types like std::string, std::vector
#include <pybind11/operators.h>
#include "SplineNetLib/SplineNet.hpp"    // Header for the library


namespace py = pybind11;

// Function to handle nested Python lists and convert them to std::vector<U>
template <typename U>
void flatten_pylist(const py::handle &obj, std::vector<U> &result) {
    if (py::isinstance<py::list>(obj)) {
        for (const auto &item : obj.cast<py::list>()) {
            flatten_pylist<U>(item, result);
        }
    } else {
        result.push_back(obj.cast<U>());
    }
}

// Wrapper function to create a new vector
template <typename U>
std::vector<U> convert_pylist_to_vector(const py::list &py_list) {
    std::vector<U> result;
    flatten_pylist<U>(py_list, result);
    return result;
}

void get_shape_recursive(const py::list& py_list, std::vector<size_t>& shape) {
    // Base case: when the list is empty, do nothing
    if (py_list.size() == 0) return;

    // Push the size of the current level
    shape.push_back(py_list.size());

    // Check if the first element is a list (nested)
    if (py::isinstance<py::list>(py_list[0])) {
        // Recursively call get_shape_recursive for nested lists
        get_shape_recursive(py::cast<py::list>(py_list[0]), shape);
    }
}

std::vector<size_t> get_shape(const py::list& py_list) {
    std::vector<size_t> shape;
    // Use the recursive get_shape implementation for vectors
    get_shape_recursive(py_list, shape);
    return shape;
}


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
        .def(py::init<unsigned int, unsigned int, unsigned int, double>())//in size, out size, detail (num of parameters -2), max (maximum input value that spline processes)
        .def(py::init<std::vector<std::vector<std::vector<std::vector<double>>>>, std::vector<std::vector<std::vector<std::vector<double>>>> >())
        .def("interpolate_splines",&SplineNetLib::layer::interpolate_splines,"None (None), calls interpolation on all splines in the layer")
        .def("forward",py::overload_cast<std::vector<double>, bool>(&SplineNetLib::layer::forward),"[double] ([double] x, bool normalize), forward call for single input sample")
        .def("forward",py::overload_cast<const std::vector<std::vector<double>> &, bool>(&SplineNetLib::layer::forward),"[[double]] (const [[double]] &x, bool normalize), forward call for batches")
        .def("backward",py::overload_cast<std::vector<double>,std::vector<double> , bool>(&SplineNetLib::layer::backward),"[double] ([double] x,[double]d_y,bool normalize), takes input x, loss gradient d_y and bool apply_grad,returns propageted loss (applies grad to all splines if True)")
        .def("backward",py::overload_cast<const std::vector<std::vector<double>> &,std::vector<std::vector<double>> >(&SplineNetLib::layer::backward),"backward but for batches (will always apply gradients)")
        .def("get_splines",&SplineNetLib::layer::get_splines,"[[SplineNetLib::spline]] (None), returns all splines in the layer");
    //int tensor
    py::class_<SplineNetLib::CTensor<int>>(m, "IntCTensor")

        .def(py::init<const std::initializer_list<int>&, const std::initializer_list<size_t>&>())
        .def(py::init<const std::vector<int>&, const std::vector<size_t>&>())
        .def(py::init<const SplineNetLib::CTensor<int>&>())
        .def(py::init([](const py::list &py_list) {//only for py module to turn nested lists and turn them to nested vector
            auto nested_vector = convert_pylist_to_vector<int>(py_list);
            std::vector<size_t> shape = get_shape(py_list);
            return new SplineNetLib::CTensor<int>(nested_vector,shape); 
        }))
        .def("data",&SplineNetLib::CTensor<int>::data,"std::vector<int>, (None), returns the stored data vector as a copy")
        .def("shape",&SplineNetLib::CTensor<int>::shape,"std::vector<size_t>, (None), returns the shape of the tensor like (dim0, dim1, ..., dimN)")
        .def("grad",&SplineNetLib::CTensor<int>::grad, "std::vector<int>, (None), returns the grad as flat 1D projected vector (internally using tensor.shape)")
        .def("zero_grad",&SplineNetLib::CTensor<int>::zero_grad, "None, (None), sets the gradient of this tensor to 0" )
        .def("squeeze",&SplineNetLib::CTensor<int>::squeeze, "None, (size_t dim), removes the dim and projects the data to the new shape")
        .def("unsqueeze",&SplineNetLib::CTensor<int>::unsqueeze, "None, (size_t dim), adds new dim at input dim index")
        .def("expand",&SplineNetLib::CTensor<int>::expand, "None, (size_t dim, size_t factor), expands the dimesnion at dim by factor -> shape: (2,2) expand(0,3) becomes: shape(6,2), (note this WILL affect the data)")
        .def("permute",&SplineNetLib::CTensor<int>::permute, "None, (std::vector<size_t>), swaps dimesnions at input indecies -> shape(2,1,3) permute([2,0,1] becomes: shape(3,2,1))")
        .def("transpose",&SplineNetLib::CTensor<int>::transpose, "None, (None), transposes the tensor (swaps the innermost two dimesnions)")
        .def("clear_history",&SplineNetLib::CTensor<int>::clear_history, "None, (None), clears all grad fns from the tensor (gradient propagatuon WILL NOT work after this so use carefully)")
        .def("clear_graph",&SplineNetLib::CTensor<int>::clear_graph,"None, (None), clears full computational graph for all tensors conected to this one")
        //.def("backward",&SplineNetLib::CTensor<int>::backward, "None, (None), backwards pass through this and connected graph")
        .def("backward", &SplineNetLib::CTensor<int>::backward, 
            py::arg("prop_grad") = std::vector<int>(), "Backward pass, takes an optional gradient vector (defaults to empty).")
        .def("__mul__", [](SplineNetLib::CTensor<int>& self, SplineNetLib::CTensor<int>& other) {return self * other;})
        .def("__add__", [](SplineNetLib::CTensor<int>& self, SplineNetLib::CTensor<int>& other) {return self + other; })
        .def("__sub__", [](SplineNetLib::CTensor<int>& self, SplineNetLib::CTensor<int>& other) {return self - other; })

        .def("__getitem__", [](SplineNetLib::CTensor<int>& self, size_t idx)->SplineNetLib::CTensor<int> { return self[idx]; });
    
    py::class_<SplineNetLib::CTensor<double>>(m, "CTensor")

        .def(py::init<const std::initializer_list<double>&, const std::initializer_list<size_t>&>())
        .def(py::init<const std::vector<double>&, const std::vector<size_t>&>())
        .def(py::init<const SplineNetLib::CTensor<double>&>())
        .def(py::init([](const py::list &py_list) {//only for py module to turn nested lists and turn them to nested vector
            auto nested_vector = convert_pylist_to_vector<double>(py_list);
            std::vector<size_t> shape = get_shape(py_list);
            return new SplineNetLib::CTensor<double>(nested_vector,shape); 
        }))
        .def("data",&SplineNetLib::CTensor<double>::data,"std::vector<int>, (None), returns the stored data vector as a copy")
        .def("shape",&SplineNetLib::CTensor<double>::shape,"std::vector<size_t>, (None), returns the shape of the tensor like (dim0, dim1, ..., dimN)")
        .def("grad",&SplineNetLib::CTensor<double>::grad, "std::vector<int>, (None), returns the grad as flat 1D projected vector (internally using tensor.shape)")
        .def("zero_grad",&SplineNetLib::CTensor<double>::zero_grad, "None, (None), sets the gradient of this tensor to 0" )
        .def("squeeze",&SplineNetLib::CTensor<double>::squeeze, "None, (size_t dim), removes the dim and projects the data to the new shape")
        .def("unsqueeze",&SplineNetLib::CTensor<double>::unsqueeze, "None, (size_t dim), adds new dim at input dim index")
        .def("expand",&SplineNetLib::CTensor<double>::expand, "None, (size_t dim, size_t factor), expands the dimesnion at dim by factor -> shape: (2,2) expand(0,3) becomes: shape(6,2), (note this WILL affect the data)")
        .def("permute",&SplineNetLib::CTensor<double>::permute, "None, (std::vector<size_t>), swaps dimesnions at input indecies -> shape(2,1,3) permute([2,0,1] becomes: shape(3,2,1))")
        .def("transpose",&SplineNetLib::CTensor<double>::transpose, "None, (None), transposes the tensor (swaps the innermost two dimesnions)")
        .def("clear_history",&SplineNetLib::CTensor<double>::clear_history, "None, (None), clears all grad fns from the tensor (gradient propagatuon WILL NOT work after this so use carefully)")
        .def("clear_graph",&SplineNetLib::CTensor<double>::clear_graph,"None, (None), clears full computational graph for all tensors conected to this one")
        //.def("backward",&SplineNetLib::CTensor<int>::backward, "None, (None), backwards pass through this and connected graph")
        .def("backward", &SplineNetLib::CTensor<double>::backward,
            py::arg("prop_grad") = std::vector<double>(),  "None, (None), backwards pass through this and connected graph")
        .def("__mul__", [](SplineNetLib::CTensor<double>& self, SplineNetLib::CTensor<double>& other) {return self * other;})
        .def("__add__", [](SplineNetLib::CTensor<double>& self, SplineNetLib::CTensor<double>& other) {return self + other; })
        .def("__sub__", [](SplineNetLib::CTensor<double>& self, SplineNetLib::CTensor<double>& other) {return self - other; })

        .def("__getitem__", [](SplineNetLib::CTensor<double>& self, size_t idx)->SplineNetLib::CTensor<double> { return self[idx]; });
        
        
}



