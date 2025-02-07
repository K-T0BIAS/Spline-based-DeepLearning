// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0




#ifndef CTENSORUTILS_HPP
#define CTENSORUTILS_HPP

#include <iostream>
#include <vector>
#include <type_traits>
#include <iterator>
#include <concepts>
#include <functional>
#include <sstream>
#include <string>
#include <any>
#include <random>
#include <unordered_set>

namespace SplineNetLib {

template <typename T>
std::string vectorToString(const std::vector<T>& vec);

template <typename T>
concept Container = requires(T t) {
    typename T::value_type;             // Requires a nested `value_type` (if T::value_type fails T is not a Container)
    typename T::iterator;               // Requires a nested `iterator`
    typename T::const_iterator;         // Also requires a nested `const_iterator` for const containers
    { t.begin() } -> std::input_iterator;  // Requires a `begin()` function that has return type std::input_iterator
    { t.end() } -> std::input_iterator;    // Requires an `end()` function that has return type std::input_iterator
    { t.size() } -> std::convertible_to<std::size_t>; //also requires a `size()` function thatvhas return type std::convertible_to<std::size_t>
};

template <typename T>
concept Scalar = std::is_arithmetic_v<T>; // Requires T to be is_arithmetic_v

// Function to generate a std::vector<T> with random values
template <Scalar T>
std::vector<T> randomVector(size_t size, T min, T max) ;


//base case for recursive n_dims check
template <Scalar T>
int get_depth(const T &scalar) ;

//Recursive case for the n_dims check will return the number of dimensions od the input
template<Container T>
int get_depth (const T &vec) ;

//base Recursive case for the get_shape func will return the shape
template <Scalar T>
std::vector<size_t> get_shape(const T &scalar, std::vector<size_t> Shape = {}) ;

//Recursive function to get shape of container (assumes uniform dims) pushes back the size of the container at current recursion depth
template <Container T>
std::vector<size_t> get_shape(const T &vec, std::vector<size_t> Shape = {}) ;

//base case if input is scalar type (will in place push back to the result)
template<typename U, Scalar T>
void Flatten(const T &in_scalar, std::vector<U> &result) ;

//Recursive case will move down one dim into the input and recursively call itself for all "values" in input
template<typename U, Container T>
void Flatten(const T &in_vector, std::vector<U> &result) ;

// Flatten controll function will create the result variable and initialize the recursion
template<typename U, typename T>
std::vector<U> Flatten(const T& in_vector) ;
    
// calculate the stride length to get to next index in dim forvthe projected vector
size_t stride(size_t idx, const std::vector<size_t> &shape) ;

//math -------------------

template<typename T>
std::vector<T> matmul(const std::vector<T> &A, const std::vector<T> &B, const std::vector<size_t> &A_shape, const std::vector<size_t> &B_shape) ;

template<typename T>
requires Scalar<T>
std::vector<T> permute_vec(const std::vector<T>& A, const std::vector<size_t>& A_shape, const std::vector<size_t>& permutation_indices) ;

} //namespace

#include "../src/CTensorUtils.tpp"

#endif