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
    
    Function(std::shared_ptr<CTensor<T>> A, std::shared_ptr<CTensor<T>> B) : a(A), b(B) {}
    
    //virtual desctructor
    virtual ~Function() = default;
    
    virtual std::vector<T> fwd() = 0;
    
    virtual void backward(std::vector<T> &prop_grad, CTensor<T> *result) = 0;
    
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
};

} //namepace

#include "../src/CTensorFunc.tpp"

#endif