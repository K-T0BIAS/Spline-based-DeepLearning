// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0




#ifndef CTENSOR_HPP
#define CTENSOR_HPP

#include "CTensorFunc.hpp"

namespace SplineNetLib {

template<typename T>
requires Scalar<T>
class Function;

template<Scalar T>
class DTensor{
public: 
    std::vector<T> _data;
    std::vector<size_t> _shape;
    std::vector<T> _grad;
    std::vector<std::unique_ptr<Function<T>>> _grad_fn;
    int _ref_c;
    
    DTensor(const std::vector<T>& data, const std::vector<size_t>& shape) : 
    _data(data), _shape(shape), _ref_c(1) {}
    
    DTensor(const std::initializer_list<T>& data, const std::initializer_list<size_t>& shape) : 
    _data(data), _shape(shape), _ref_c(1) {}
    
    DTensor(const DTensor<T>& other) : _data(other._data), _shape(other._shape), _grad(other._grad), _ref_c(1) {
        // Deep copy unique_ptrs to grad fns by calling clone()
        for (const auto& fn : other._grad_fn) {
            _grad_fn.push_back(fn ? fn->clone() : nullptr);
        }
    }
    
    void add_ref(){
        _ref_c++;
    }
    
    void rmf_ref(){
        _ref_c--;
        if (_ref_c == 0){
            delete this;
        }
    }
};


template<Scalar T>
class CTensor { 
private:
    
    CTensor(DTensor<T>* _t_data) : _tensor_data(_t_data){}
    
public:

    DTensor<T>* _tensor_data;
    
    bool requires_grad = true;
        
    CTensor(const std::initializer_list<T>& init, const std::initializer_list<size_t>& shape) {
        _tensor_data = new DTensor(init, shape);
    }
    
    
    CTensor(const std::vector<T>& data, const std::vector<size_t>& shape) {
        _tensor_data = new DTensor(data, shape);
    }
    
    template<Container U>
    CTensor(const U& data) {
        _tensor_data = new DTensor(Flatten<T>(data), get_shape(data));
    }
    
    CTensor(const CTensor<T>& other){
        _tensor_data = other._tensor_data;
        _tensor_data->_ref_c++;
    }
    
    
    
    
    ~CTensor(){
        _tensor_data->rmf_ref();
    }
    
    //-----getters-----
    
    std::vector<T> data() const { return this->_tensor_data->_data; }
    
    std::vector<size_t> shape() const { return this->_tensor_data->_shape; }
    
    std::vector<T> grad() const { return this->_tensor_data->_grad; }
    
    std::vector<std::unique_ptr<Function<T>>> grad_fn() const { return this->_tensor_data->grad_fn; }
    
    void zero_grad();
    
    CTensor<T> clone();
    
    //-----shape-utils-----
    
    void squeeze(const size_t &dim) ;//squeezes / removes the input dim and changes the internal projection shape
    
    void unsqueeze(const size_t &dim) ; //adds a new dim at the input dim
    
    void expand(const size_t &dim, const size_t &factor) ; //expands the dimension by factor so that shape 3,2 expanded(1,3) 
                                                           //becomes: (3,6) (will duplicate values at the dimension to match new projected shape)
    
    void permute(const std::vector<size_t> &permutation_indecies) ; //will swap dimesnions at the permutation indecies 
                                                                    //shape (2,3,4) permute(2,0,1) becomes: (4,2,3) 
    
    void transpose() ;
    
    //-----auto_grad-----
    //delete all grad fns of this 
    void clear_history() ;
    //recursive delete of grad fns for all tensors in the graph with this as root
    void clear_graph() ;
    //maybe add overload o this so that f no arg was passed propagated grad is set to {}, than this function below could use all by ref
    void backward(std::vector<T> prop_grad = {}) ;
    
    
    //-----operator-----
    
    auto operator[](size_t idx) ;
    
    auto operator+(CTensor<T> &other) ;
    
    auto operator-(CTensor<T> &other) ;
    
    auto operator*(CTensor<T> &other) ;
    
    //CTensor<T>& operator=(const CTensor<T> &other) noexcept;
    
    //CTensor<T>& operator=(CTensor<T> &&other) ;

    
};
/*
template<Scalar T>
CTensor<T> zeros(std::vector<size_t> shape) ;

template<Scalar T>
CTensor<T> ones(std::vector<size_t> shape) ;

template<Scalar T>
CTensor<T> random(std::vector<size_t> shape, T min, T max) ;

template<typename T ,Container U>
CTensor<T> Tensor(U data) ;

template<typename T, Scalar U>
CTensor<T> Tensor(U data) ;

template<typename T>
CTensor<T> Tensor(std::vector<T> data, std::vector<size_t> shape) ;
*/
} //namespace

#include "../src/CTensor.tpp"


#endif