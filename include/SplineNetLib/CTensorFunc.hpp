// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0




#ifndef CTENSORFUNC_HPP
#define CTENSORFUNC_HPP

#include "CTensorUtils.hpp"

namespace SplineNetLib {
    
typedef enum {
    RESHAPE_SQUEEZE = 1,
    RESHAPE_UNSQUEEZE  = 2,
    RESHAPE_EXPAND = 3,
    RESHAPE_REDUCE = 4,
    RESHAPE_PERMUTE = 5,
    RESHAPE_TRANSPOSE = 6
} ReshapeType;


template<Scalar T>
class CTensor;

//base function class for specialization
template<typename T>
requires Scalar<T>
class Function {
public:
    //pointers to this functions "parents" (like : a operator b)
    std::shared_ptr<CTensor<T>> a;
    std::shared_ptr<CTensor<T>> b;
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    
    Function(std::shared_ptr<CTensor<T>> A, std::shared_ptr<CTensor<T>> B) : a(A), b(B), 
               /*nullptr check for A and B to ensure no segfaults happen ->*/a_shape(A ? A->shape() : std::vector<size_t> {1}), 
                                                                             b_shape(B ? B->shape() : std::vector<size_t> {1}) {}
    
    //virtual desctructor
    virtual ~Function() = default;
    
    virtual std::vector<T> fwd() = 0;
    
    virtual void backward(std::vector<T> &prop_grad, CTensor<T> *result) = 0;
    
    virtual std::unique_ptr<Function<T>> clone() const = 0;
    
    static std::unordered_set<Function<T>*> global_chain;
    
    void clear_graph_f();
};

template<typename T>
requires Scalar<T>
std::unordered_set<Function<T>*> Function<T>::global_chain;

//addition class for CTensor<T>::operator+
template<typename T>
requires Scalar<T>
class AddFunction : public Function<T> {
public:

    //construct base class
    AddFunction(std::shared_ptr<CTensor<T>> a, std::shared_ptr<CTensor<T>> b) : Function<T>(a, b) {}
    
    std::vector<T> fwd() override ;
    
    void backward(std::vector<T> &prop_grad, CTensor<T> *result) override;
    
    virtual std::unique_ptr<Function<T>> clone() const override;
};

//subtractor function class for CTensor<T>::operator-
template<typename T>
requires Scalar<T>
class SubFunction : public Function<T> {
public:

    //construct base class
    SubFunction(std::shared_ptr<CTensor<T>> a, std::shared_ptr<CTensor<T>> b) : Function<T>(a, b) {}
    
    std::vector<T> fwd() override;
    
    void backward(std::vector<T> &prop_grad, CTensor<T> *result) override;
    
    virtual std::unique_ptr<Function<T>> clone() const override;
    
};

//matrix multiplication function class for CTensor<T>::operator*
template<typename T>
requires Scalar<T>
class MatMulFunction : public Function<T> {
public:

    //construct base class
    MatMulFunction(std::shared_ptr<CTensor<T>> a, std::shared_ptr<CTensor<T>> b) : Function<T>(a, b) {}
    
    std::vector<T> fwd() override;
    
    void backward(std::vector<T> &prop_grad, CTensor<T> *result) override;
    
    virtual std::unique_ptr<Function<T>> clone() const override;
};

template<typename T>
requires Scalar<T>
class ReShapeFunction : public Function<T> {
public :
    
    ReshapeType operation;
    /*
    std::vector<size_t> original_shape;
    std::vector<size_t> new_shape;
    */
    
    ReShapeFunction(std::shared_ptr<CTensor<T>> a, ReshapeType _operation) : 
    Function<T>(a, nullptr),operation(_operation){}
    
    std::vector<T> fwd() override;
    
    void backward(std::vector<T> &prop_grad, CTensor<T> *result) override;
    
    virtual std::unique_ptr<Function<T>> clone() const override;
};

} //namepace

#include "../src/CTensorFunc.tpp"

#endif