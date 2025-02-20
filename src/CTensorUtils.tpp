// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0




#ifndef CTENSORUTILS_TPP
#define CTENSORUTILS_TPP

#include "../include/SplineNetLib/CTensorUtils.hpp"

namespace SplineNetLib {

template <Scalar T>
std::vector<T> randomVector(size_t size, T min, T max) {
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Distribution depending on type T
    typename std::conditional<std::is_integral<T>::value, 
                              std::uniform_int_distribution<T>, 
                              std::uniform_real_distribution<T>>::type dist(min, max);

    std::vector<T> vec(size);
    for (auto& v : vec) {
        v = dist(gen);
    }
    return vec;
}


template <Scalar T>
int get_depth(const T &scalar) {
    return 0;
}

template <Container T>
int get_depth(const T &vec) {
    int max_depth = 1;
    for (const auto &element : vec) {
        max_depth = std::max(max_depth, 1 + get_depth(element));
    }
    return max_depth;
}

template<typename T>
std::string vectorToString(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i < vec.size() - 1) {
            oss << ", ";
        }
    }
    oss << ")";
    return oss.str();
}

template <Scalar T>
std::vector<size_t> get_shape(const T &scalar, std::vector<size_t> Shape) {
    return Shape;
}

template <Container T>
std::vector<size_t> get_shape(const T &vec, std::vector<size_t> Shape) {
    Shape.push_back(vec.size());
    return get_shape(vec[0], Shape);
}

template<typename U, Scalar T>
void Flatten(const T &in_scalar, std::vector<U> &result) {
    result.push_back(in_scalar);
}

template<typename U, Container T>
void Flatten(const T &in_vector, std::vector<U> &result) {
    for (const auto &vec : in_vector) {
        Flatten(vec, result);
    }
}

template<typename U, typename T>
std::vector<U> Flatten(const T& in_vector) {
    std::vector<U> result;
    Flatten(in_vector, result);
    return result;
}

inline size_t stride(size_t idx, const std::vector<size_t> &shape) {
    size_t stride = 1;
    for (size_t i = idx + 1; i < shape.size(); i++) {
        stride *= shape[i];
    }
    return stride;
}

//math funcs

template<typename T>  // Template function that accepts any scalar type 'T' (e.g., float, double)
requires Scalar<T>   // This constraint ensures that the type 'T' is a scalar (e.g., not a matrix, vector, etc.)
std::vector<T> matmul(const std::vector<T> &A, const std::vector<T> &B, const std::vector<size_t> &A_shape, const std::vector<size_t> &B_shape) {
    size_t batch_size = 1;  // Variable to store the number of batches (default to 1)
    //std::cout<<"debug : matmul : a shape = "<<vectorToString(A_shape)<<" b shape = "<<vectorToString(B_shape)<<"\n";
    // Ensure A and B have the same number of dimensions
    if (B_shape.size() != A_shape.size()) {
        throw std::invalid_argument("A_shape.size() and B_shape.size() must be equal");
        return std::vector<T>(1, 0);  // This return statement is unreachable due to the exception, but just in case.
    }
    
    
    // If A has more than 2 dimensions (e.g., batching is involved), calculate the batch size
    if (A_shape.size() > 2) {
        for (size_t i = 0; i < A_shape.size() - 2; i++) {
            batch_size *= A_shape[i];  // Multiply the sizes of the leading dimensions (batch dimensions)
        }
    }

    // Get the dimensions for matrix multiplication
    size_t M = A_shape[A_shape.size() - 2];  // Rows of A
    size_t K = A_shape[A_shape.size() - 1];  // Columns of A and rows of B
    size_t N = B_shape[B_shape.size() - 1];  // Columns of B

    // Initialize the result vector with a size to hold all results (batch_size * M * N)
    std::vector<T> result(batch_size * M * N);

    // Perform matrix multiplication for each batch
    for (size_t batch_dim = 0; batch_dim < batch_size; batch_dim++) {
        for (size_t row = 0; row < M; row++) {  // Iterate over each row of A
            for (size_t col = 0; col < N; col++) {  // Iterate over each column of B
                T sum = 0.0;  // Initialize the sum for the current element in the result matrix
                for (size_t shared = 0; shared < K; shared++) {  // Iterate over the shared dimension (columns of A, rows of B)
                    // Perform the dot product between the row of A and the column of B
                    sum += A[batch_dim * M * K + row * K + shared] * B[batch_dim * K * N + shared * N + col];
                }
                // Store the computed value in the result vector at the appropriate position
                result[batch_dim * M * N + row * N + col] = sum;
            }
        }
    }
    return result;  // Return the final result of the matrix multiplication
}

template<typename T>
requires Scalar<T>
std::vector<T> permute_vec(const std::vector<T>& A, const std::vector<size_t>& A_shape, const std::vector<size_t>& permutation_indices) {
    std::vector<T> B(A.size(), 0);
    std::vector<size_t> B_shape;

    for (const auto& idx : permutation_indices) {
        B_shape.push_back(A_shape[idx]);
    }

    for (size_t i = 0; i < A.size(); i++) {
        size_t idx = 0;
        for (size_t k = 0; k < A_shape.size(); k++) {
            idx += ((i / stride(permutation_indices[k], A_shape)) % B_shape[k]) * stride(k, B_shape);
        }
        B[idx] = A[i];
    }
    return B;
}

inline std::vector<size_t> transpose_shape(const std::vector<size_t>& shape) {
    std::vector<size_t> temp = shape;
    size_t n_dims = temp.size();
    temp[n_dims - 2] = shape[n_dims - 1];
    temp[n_dims - 1] = shape[n_dims - 2];
    return temp;
}

}//namespace

#endif