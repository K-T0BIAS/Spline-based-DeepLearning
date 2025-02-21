// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0




#ifndef CTENSORFUNC_TPP
#define CTENSORFUNC_TPP


#include "../include/SplineNetLib/CTensorFunc.hpp"

namespace SplineNetLib {

template<typename T>
requires Scalar<T>
void Function<T>::clear_graph_f() {
    a->clear_graph();
    b->clear_graph();
}

template<typename T>
requires Scalar<T>
std::vector<T> AddFunction<T>::fwd() {
    
    auto* a_data = &(this->a->_tensor_data->_data);
    auto* b_data = &(this->b->_tensor_data->_data);
    
    T l;
    T r;
    
    std::vector<T> res_vec;
    for (size_t i = 0; i < (*a_data).size() || i < (*b_data).size(); i++){
        l = (i < (*a_data).size()) ? (*a_data)[i] : 0 ;
        r = (i < (*b_data).size()) ? (*b_data)[i] : 0 ;        
        res_vec.push_back(l + r);
    }
    return res_vec;
}
    


template<typename T>
requires Scalar<T>
void AddFunction<T>::backward(std::vector<T> &prop_grad, CTensor<T> *result) {
    //std::cout<<"debug add bwd call\n";
    //check if func already exists in the recursive chain
    if (Function<T>::global_chain.find(this) != Function<T>::global_chain.end()) {
        std::cout<<"cyle detected in grad backward, ensure no incorrect reassignments to to Ctensors that were previously used in the computation graph\n";
        return;
    }
    //std::cout<<"debug add bwd cycle check\n";
    //insert this func into the chain for cycle detection
    Function<T>::global_chain.insert(this);
    //std::cout<<"debug add bwd chain insert\n";
    if (prop_grad.empty()){
        for (size_t i=0; i < this->a->data().size(); i++) {
            prop_grad.push_back(1);
            //std::cout<<"debug add bwd empty grad set to 1s \n";
        }
    }
    //std::cout<<"debug add bwd grad add\n";
    //ensure self dependend gradients arent added twice
    if (result != this->a.get()) {
        //std::cout<<"debug add bwd this->a gradient propagation initialized\n";
        //std::cout<<"debug add bwd this a grad size:"<<this->a->grad().size()<<"prop_grad size: "<<prop_grad.size()<<"\n";
        if (this->a->requires_grad == true) {
            if (this->a->grad().empty()){
                //std::cout<<"a grqd empty "<<this->a->grad().size()<<"\n";
                this->a->zero_grad();
            }
            //std::cout<<"working on grad of a at "<<this->a<<" "<<vectorToString(this->a->grad())<<" "<<vectorToString(prop_grad)<<"\n";
            for (size_t i = 0; i < prop_grad.size(); i++) {
                
                this->a->_tensor_data->_grad[i] += prop_grad[i];
                //std::cout<<"debug add bwd accumulation step\n";
            }
        }
        //std::cout<<"debug add bwd this a grad accumulated\n";
        this->a->backward(prop_grad);
        //std::cout<<"debug add bwd this a recursion finished\n";
    }
    //ensure self dependend gradients arent added twice
    if (result != this->b.get()) {
        //std::cout<<"debug add bwd this->b gradient propagation initialized\n";
        //std::cout<<"debug add bwd this b grad size:"<<this->b->grad().size()<<"prop_grad size: "<<prop_grad.size()<<"\n";
        if (this->b->requires_grad == true) {
            if (this->b->grad().empty()){
                //std::cout<<"b grqd empty "<<this->b->grad().size()<<"\n";
                this->b->zero_grad();
            }
            //std::cout<<"working on grad of b at "<<this->b<<" "<<vectorToString(this->b->grad())<<" "<<vectorToString(prop_grad)<<"\n";
            for (size_t i = 0; i < prop_grad.size(); i++) {
                
                this->b->_tensor_data->_grad[i] += prop_grad[i];
                //std::cout<<"debug add bwd accumulation step\n";
            }
        }
        //std::cout<<"debug add bwd this b grad accumulated\n";
        this->b->backward(prop_grad);
        //std::cout<<"debug add bwd this b recursion finished\n";
    }
    //std::cout<<"debug add bwd recursive propagation\n";
    //remove this func from the chain if all its recursive processes finished
    Function<T>::global_chain.erase(this);
    //std::cout<<"debug add bwd chain erase\n";
}

template<typename T>
requires Scalar<T>
std::unique_ptr<Function<T>> AddFunction<T>::clone() const {
    return std::make_unique<AddFunction<T>>(*this);
}


template<typename T>
requires Scalar<T>
std::vector<T> SubFunction<T>::fwd() {
    
    auto* a_data = &(this->a->_tensor_data->_data);
    auto* b_data = &(this->b->_tensor_data->_data);
    
    T l;
    T r;
    
    std::vector<T> res_vec;
    for (size_t i = 0; i < (*a_data).size() || i < (*b_data).size(); i++){
        l = (i < (*a_data).size()) ? (*a_data)[i] : 0 ;
        r = (i < (*b_data).size()) ? (*b_data)[i] : 0 ;        
        res_vec.push_back(l - r);
    }
    return res_vec;
}


template<typename T>
requires Scalar<T>
void SubFunction<T>::backward(std::vector<T> &prop_grad, CTensor<T> *result) {
    
    //check if func already exists in the recursive chain
    if (Function<T>::global_chain.find(this) != Function<T>::global_chain.end()) {
        std::cout<<"cyle detected in  Ctensor.backward(), ensure no incorrect reassignments to to Ctensors that were previously used in the computation graph\n";
        return;
    }
    
    //insert this func into the chain for cycle detection
    Function<T>::global_chain.insert(this);
        
    if (prop_grad.empty()){
        for (size_t i=0; i < this->a->data().size(); i++) {
            prop_grad.push_back(1);
        }
    }
    
    //ensure self dependend gradients arent added twice
    if (result != this->a.get()) {
        if (this->a->requires_grad == true) {
            if (this->a->grad().empty()){
                //std::cout<<"a grqd empty "<<this->a->grad().size()<<"\n";
                this->a->zero_grad();
            }
            for (size_t i = 0; i < prop_grad.size(); i++) {
                
                this->a->_tensor_data->_grad[i] += prop_grad[i];
                
            }
        }
        this->a->backward(prop_grad);
    }
    //ensure self dependend gradients arent added twice
    if (result != this->b.get()) {
        if (this->b->requires_grad == true) {
            if (this->b->grad().empty()){
                //std::cout<<"b grqd empty "<<this->b->grad().size()<<"\n";
                this->b->zero_grad();
            }
            for (size_t i = 0; i < prop_grad.size(); i++) {
                this->b->_tensor_data->_grad[i] -= prop_grad[i];
            }
        }
        this->b->backward(prop_grad);
    }
    //remove this func from the chain if all its recursive processes finished
    Function<T>::global_chain.erase(this);
}


template<typename T>
requires Scalar<T>
std::unique_ptr<Function<T>> SubFunction<T>::clone() const {
    return std::make_unique<SubFunction<T>>(*this);
}

template<typename T>
requires Scalar<T>
std::vector<T> MatMulFunction<T>::fwd() {
    
    std::vector<size_t> a_shape = this->a->shape();
    std::vector<size_t> b_shape = this->b->shape();
        
    size_t a_n_dims = a_shape.size();
    size_t b_n_dims = b_shape.size();
    
    auto a_copy = this->a->clone();
    auto b_copy = this->b->clone();
        
    if (a_n_dims != b_n_dims) {
        throw std::invalid_argument("operator (*) expects both opperants to have the same num of dimensions but got:"+std::to_string(a_n_dims)+"and "+std::to_string(b_n_dims)+",please ensure opperants dims match by using squeeze or unsqueeze beforehand\n");
    }
    if (a_n_dims > 2) {
        //Create sub vectors for the batch dimensions
        std::vector<size_t> a_batch_shape;
        std::vector<size_t> b_batch_shape;
        //get only the batch dimension shapes
        for (size_t i = 0; i < a_shape.size()-2; i++ ){
            a_batch_shape.push_back(a_shape[i]);
            b_batch_shape.push_back(b_shape[i]);
        }
        for (size_t i = 0; i < a_batch_shape.size(); i++) {
            //expand dims so that batch dimensions are the same
            if (a_batch_shape[i] != b_batch_shape[i]) {
                a_copy.expand(i,b_batch_shape[i]);
                b_copy.expand(i,a_batch_shape[i]);
            }
        }
    }
    std::vector<T> result_vector = matmul(a_copy.data(), b_copy.data(), a_copy.shape(), b_copy.shape());
    return result_vector;
    
}

template<typename T>
requires Scalar<T>
void MatMulFunction<T>::backward(std::vector<T> &prop_grad, CTensor<T> *result) {
    
    //check if func already exists in the recursive chain
    if (Function<T>::global_chain.find(this) != Function<T>::global_chain.end()) {
        std::cout<<"cyle detected in  Ctensor.backward(), ensure no incorrect reassignments to to Ctensors that were previously used in the computation graph\n";
        return;
    }
    
        //insert this func into the chain for cycle detection
    Function<T>::global_chain.insert(this);

    
    auto prop_grad_shape = result->shape();
    //std::cout<<"matmul bwd prop shape : "<<vectorToString(prop_grad_shape)<<"\n";
    

    if (prop_grad.empty()){
        for (size_t i=0; i < result->data().size(); i++) {
            prop_grad.push_back(1);
        }
    }
    
    //ensure self dependend gradients arent added twice
    if (result != this->a.get()) {
        auto prop_grad_a = this->a->grad(); //needs to be deeply checked
        if (this->a->requires_grad == true) {
            if (this->a->_tensor_data->_grad.empty()){
                //std::cout<<"a grqd empty "<<this->a->grad().size()<<"\n";
                this->a->zero_grad();
            }
                    //create a copy of b and transpose it
            auto b_copy = this->b->clone();
            b_copy.transpose();
            auto b_shape = transpose_shape(this->b_shape);
            
            prop_grad_a = matmul(prop_grad, b_copy.data(), prop_grad_shape, b_shape);
            
            //assign grad
            for (size_t i = 0; i < prop_grad_a.size(); i++) {
                this->a->_tensor_data->_grad[i] += prop_grad_a[i];
            }
        }
        this->a->backward(prop_grad_a);
    }
    
    //ensure self dependend gradients arent added twice
    if (result != this->b.get()) {
        auto prop_grad_b = this->b->grad();
        if (this->b->requires_grad == true) {
            if (this->b->_tensor_data->_grad.empty()){
                //std::cout<<"b grad empty "<<this->b->grad().size()<<"\n";
                this->b->zero_grad();
            }
                    //create a copy of b and transpose it
            auto a_copy = this->a->clone();
            a_copy.transpose();
            auto a_shape = transpose_shape(this->a_shape);
            //std::cout<<"b bwd a_copy shape :"<<vectorToString(a_copy.shape())<<" "<<vectorToString(prop_grad)<<"\n";
            prop_grad_b = matmul(a_copy.data(), prop_grad, a_shape, prop_grad_shape);
            
            //assign grad
            for (size_t i = 0; i < prop_grad_b.size(); i++) {
                this->b->_tensor_data->_grad[i] += prop_grad_b[i];
            }
        }
        this->b->backward(prop_grad_b);
    }
    
    //remove this func from the chain if all its recursive processes finished
    Function<T>::global_chain.erase(this);
}

template<typename T>
requires Scalar<T>
std::unique_ptr<Function<T>> MatMulFunction<T>::clone() const {
    return std::make_unique<MatMulFunction<T>>(*this);
}

template<typename T>
requires Scalar<T>
std::vector<T> ReShapeFunction<T>::fwd() {
    return this->a->data();
}


template<typename T>
requires Scalar<T>
void ReShapeFunction<T>::backward(std::vector<T> &prop_grad, CTensor<T> *result){
    //std::cout<<"RESHAPEFUNCTION CALL\n";
    
    switch(this->operation) {
        case RESHAPE_SQUEEZE:
            if (result != this->a.get()){
                this->a->backward(prop_grad);
            }
            break;
        case RESHAPE_UNSQUEEZE:
            if (result != this->a.get()){
                this->a->backward(prop_grad);
            }
            break;
        case RESHAPE_EXPAND: 
            std::cout<<"\n\nWARNING: This CTensor was expanded in the computational graph, therefore gradients can not be calculated further in this branch\n\n";
            break;
        
        case RESHAPE_REDUCE:
            std::cout<<"\n\nWARNING: This CTensor was reduced in the computational graph, therefore gradients can not be calculated further in this branch\n\n";
            break;
        case RESHAPE_PERMUTE:
            if (result != this->a.get()){
                this->a->backward(prop_grad);
            }
            break;
        case RESHAPE_TRANSPOSE:
            if (result != this->a.get()){
                this->a->backward(prop_grad);
            }
            break;
        default: //should throw exeption
            throw std::runtime_error("\nFound unknown grad_fn type during backward propagation\n");
            break;
    }
}

template<typename T>
requires Scalar<T>
std::unique_ptr<Function<T>> ReShapeFunction<T>::clone() const{
    return std::make_unique<ReShapeFunction<T>>(*this);
}

}//namespace

#endif