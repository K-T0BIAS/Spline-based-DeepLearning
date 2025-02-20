// Copyright (c) <2025>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0





#ifndef CTENSOR_TPP
#define CTENSOR_TPP


#include "../include/SplineNetLib/CTensor.hpp"

namespace SplineNetLib {

template<Scalar T>
void CTensor<T>::zero_grad(){
    this->_tensor_data->_grad = std::vector(this->_tensor_data->_data.size(),static_cast<T>(0));
}


template<Scalar T>
void CTensor<T>::squeeze(const size_t& dim) {
    auto n_dims = this->_tensor_data->_shape.size();
    if (n_dims == 1) {
        throw std::invalid_argument("CTensor with 1 Dim can not be squeezed to be 0D\n");
    } else if (dim >= n_dims) {
        throw std::invalid_argument("target Dim: "+std::to_string(dim)+"is out of range of CTensor with n_dims: "+std::to_string(n_dims)+"\n");
    } else if (dim == n_dims-1){
        this->_tensor_data->_shape[dim-1] *= this->_tensor_data->_shape[dim];
        this->_tensor_data->_shape.pop_back();
    } else {
        this->_tensor_data->_shape[dim] *= this->_tensor_data->_shape[dim+1];
        this->_tensor_data->_shape.erase(this->_tensor_data->_shape.begin() + dim + 1);
    }
    
    if (this->requires_grad) {
        auto new_fn = std::make_unique<ReShapeFunction<T>>(std::make_shared<CTensor<T>>(*this), RESHAPE_SQUEEZE);
        
        this->_tensor_data->_grad_fn.push_back(std::move(new_fn));
    }
}
    
template<Scalar T>
void CTensor<T>::unsqueeze(const size_t &dim) {
    auto n_dims = this->_tensor_data->_shape.size();
    auto* shape = &(this->_tensor_data->_shape);//make a temp ptr to the shape vector for easier syntax
    if (dim > n_dims) {
        (*shape).push_back(1);
    } else {
        (*shape).insert((*shape).begin() + dim, 1);
    }
    
    if (this->requires_grad) {
        auto new_fn = std::make_unique<ReShapeFunction<T>>(std::make_shared<CTensor<T>>(*this), RESHAPE_UNSQUEEZE);
        
        this->_tensor_data->_grad_fn.push_back(std::move(new_fn));
    }
}

template<Scalar T>
void CTensor<T>::expand(const size_t &dim, const size_t &factor) {
    if (factor <= 1) {
        return; // No expansion needed
    }
    
    auto* shape = &(this->_tensor_data->_shape);//make a temp ptr to the shape vector for easier syntax
    auto* data = &(this->_tensor_data->_data);
    auto n_dims = (*shape).size();
    
    
    // Check if the specified dimension is valid
    if (dim >= n_dims) {
        throw std::invalid_argument("Input dim: " + std::to_string(dim) + " cannot be larger than _n_dims: " + std::to_string(n_dims));
    }
    
    // Calculate the size of the sub-vectors to repeat
    size_t sub_vector_size = 1;
    for (size_t i = dim + 1; i < n_dims; i++) {
        sub_vector_size *= (*shape)[i];
    }
    
    size_t data_size_per_expansion = (*shape)[dim] * sub_vector_size;
        
    // Repeat the data by the specified factor
    size_t idx = 0;
    while (idx < (*data).size()) {
        std::vector<T> sub_vector((*data).begin() + idx, (*data).begin() + idx + data_size_per_expansion);
    
        // Insert the sub-vector factor times
        for (size_t i = 1; i < factor; i++) {
            (*data).insert((*data).begin() + idx, sub_vector.begin(), sub_vector.end());
            idx += data_size_per_expansion;
        }
            
        idx += data_size_per_expansion;
    }
    
    auto new_shape = (*shape);
    new_shape[dim] *= factor;
    
        //create new addfunction with shared ptr to this and other
    auto new_fn = std::make_unique<ReShapeFunction<T>>(std::make_shared<CTensor<T>>(*this), RESHAPE_EXPAND);
    
        // Update the shape and number of dimensions
    (*shape)[dim] *= factor;
    
    this->_tensor_data->_grad_fn.push_back(std::move(new_fn));

}

template <Scalar T>
void CTensor<T>::reduce(const size_t &dim, const size_t &factor) {
    if (factor <= 1) {
        return; // No reduction needed
    }

    auto* shape = &(this->_tensor_data->_shape); // Pointer to shape vector
    auto* data = &(this->_tensor_data->_data);
    size_t n_dims = shape->size();

    // Ensure valid dimension
    if (dim >= n_dims) {
        throw std::invalid_argument("Input dim: " + std::to_string(dim) + 
                                    " cannot be larger than _n_dims: " + std::to_string(n_dims));
    }

    // Ensure the shape is divisible by factor
    if ((*shape)[dim] % factor != 0) {
        return;
    }

    // Calculate the size of sub-vectors
    size_t sub_vector_size = 1;
    for (size_t i = dim + 1; i < n_dims; i++) {
        sub_vector_size *= (*shape)[i];
    }

    size_t idx = 0;
    while (idx < data->size()) {
        // Remove (factor - 1) repetitions of the sub-vector
        for (size_t i = 1; i < factor; i++) {
            data->erase(data->begin() + idx, data->begin() + idx + sub_vector_size);
        }
        idx += sub_vector_size;  // Move to the next section after all removals
    }

    (*shape)[dim] /= factor;
}

template<Scalar T>
void CTensor<T>::permute(const std::vector<size_t> &permutation_indecies) {
    //renamed global func permute to permute_vec so that func czll in class is nolonger ::permute now permute_vec
    this->_tensor_data->_data = permute_vec(this->_tensor_data->_data, this->_tensor_data->_shape, permutation_indecies); 
        
    auto shape_copy = this->shape();
    for (size_t i = 0; i < permutation_indecies.size(); i++) {
        this->_tensor_data->_shape[i] = shape_copy[permutation_indecies[i]];
    }
}

template<Scalar T>
void CTensor<T>::transpose() {
    if (this->_tensor_data->_shape.size()>=2) {
        std::vector<size_t> transpose_idx;
        for (size_t i = 0; i < this->_tensor_data->_shape.size()-2; i++) {
            transpose_idx.push_back(i);
        }
            
        transpose_idx.push_back(this->_tensor_data->_shape.size() - 1);
        transpose_idx.push_back(this->_tensor_data->_shape.size() - 2);
            
        this->permute(transpose_idx);
    } 
}



//-----operator-----/
template<Scalar T>
auto CTensor<T>::operator[](size_t idx){
    std::vector<size_t> Shape = this->shape();
    //check if index should exist in multi dim space
    if (idx >= Shape[0]) {
        throw std::invalid_argument("index ["+std::to_string(idx)+"] is out of range with dim of size : "+std::to_string(Shape[0])+"\n");
    }
    //if vector is 1D to begin with
    if (Shape.size() == 1) {
        //create sub vector with scalar data at data[idx]
        std::vector<typename decltype(this->data())::value_type> sub_vector = {this->data()[idx]};
        //std::cout<<"operator[] scalar case debug data[idx]="<<sub_vector[0]<<"\n"; //debug
        Shape = {1};  // now just a scalar (still packed in a vector but treated as scalar)
        return CTensor(sub_vector, Shape);
    }
            
    //remove first dim from Shape as sub vector only uses the later dims
    Shape.erase(Shape.begin());
    size_t size_sub_vector = 1;
    //calculate the projected size of the sub tensor in 1D spcae
    for (const size_t& dim : Shape) {
        size_sub_vector *= dim;
    }
            
    //projected index to 1D
    size_t flat_idx = idx * size_sub_vector;
    //to avoid exessivevcalls to this->data() in range constructor (could likely also be used in decltype)
    auto data = this->data();
    //creates a vector of same type as stored in CTensor using range constructor from flat_idx to flat_idx + size_sub_vector
    std::vector<typename decltype(this->data())::value_type> sub_vector(data.begin() + flat_idx, data.begin() + flat_idx + size_sub_vector);
        
        
    auto new_CT = CTensor(sub_vector, Shape);
    return new_CT;
}

template<Scalar T>
auto CTensor<T>::operator+(CTensor<T>& other){
    //create new addfunction with shared ptr to this and other
    auto new_fn = std::make_unique<AddFunction<T>>(std::make_shared<CTensor<T>>(*this),
                                                    std::make_shared<CTensor<T>>(other));
    auto res_vec = new_fn->fwd(); //add this data and other data
    auto result = CTensor(res_vec, this->shape());//create the result CTensor 
    if (this->requires_grad || other.requires_grad) {
        result.requires_grad = true;
        
        result._tensor_data->_grad_fn.push_back(std::move(new_fn));
    } else {
        result.requires_grad = false;
    }
    return result;
}


template<Scalar T>
auto CTensor<T>::operator-(CTensor<T> &other) {
    //create new SubFunction with shared ptr to this and other
    auto new_fn = std::make_unique<SubFunction<T>>(std::make_shared<CTensor<T>>(*this),
                                                   std::make_shared<CTensor<T>>(other));
    auto res_vec = new_fn->fwd();
    auto result = CTensor<T>(res_vec, this->shape());
    if (this->requires_grad || other.requires_grad) {
        result.requires_grad = true;
        
        result._tensor_data->_grad_fn.push_back(std::move(new_fn));
    } else {
        result.requires_grad = false;
    }
    return result;
}

template<Scalar T>
auto CTensor<T>::operator* (CTensor<T> &other) {
    //create the parent function for the result using parents this and other
    //this will make a shared ptr of the base class. this works since the functions in tje derived classes are all overrides 
    //this is doen so that all grad fns of a CTensor can be stored in the same std::vector<shared_ptr<Function<T>>> _grad_fn
    auto new_fn = std::make_unique<MatMulFunction<T>>(std::make_shared<CTensor<T>>(*this),
                                                     std::make_shared<CTensor<T>>(other));
    //use new_fn.forward() to perfo5m the addition
    auto res_vec = new_fn->fwd();
    
    std::vector<size_t> result_shape;
    auto this_shape = this->shape();
    auto other_shape = other.shape();
    for (size_t i = 0; i < this_shape.size() -1; i++){
        result_shape.push_back(this_shape[i]);
    }
    result_shape.push_back(other_shape[other_shape.size()-1]);
        
    auto result = CTensor<T>(res_vec, result_shape);
    //assign parent function to the result._grad_fn
    if (this->requires_grad || other.requires_grad) {
        result.requires_grad = true;
        
        result._tensor_data->_grad_fn.push_back(std::move(new_fn));
    } else {
        result.requires_grad = false;
    }
    
    return result;
}





template<Scalar T>
void CTensor<T>::clear_history() {
    this->_tensor_data->_grad_fn.clear();
    //this should be safe since Function uses pointers to Ctensor and the Tensor will survive the _grad_fn clear
}

template<Scalar T>
void CTensor<T>::clear_graph() {
    //recursive call to traverse grad graph 
    for (auto &fn : this->_tensor_data->_grad_fn) {

        fn->clear_graph_f();
    }
    //clear this CTensor history when sub graph is cleared
    this->clear_history();
}

//can be improved with overload if no arg is passe to use {} so that this function below can use refernces
template<Scalar T>
void CTensor<T>::backward(std::vector<T> prop_grad) {
    /*
    //go through all parent Functions
    for (auto &fn : this->_tensor_data->_grad_fn) {
        if (fn) {
            //std::cout<<sizeof(fn)<<"\n";
            //cast fn to Function type (not done before bc circular dependencies between Ctensor and Function)
            //call backward on fn and pass prop_grad as propagated gradient and 'this' as result ptr (result = child of parent func)
            //std::cout<<"debug Ct bwd fn bwd call\n";
            fn->backward(prop_grad, this);
            //std::cout<<"debug Ct bwd fn bwd finish\n";
        }
    }
    //std::cout<<"debug Ct bwd fn all bwd finish\n";
    */
    //testing with revers as this makes more sense fir the tree traversal
    for (int i = this->_tensor_data->_grad_fn.size() - 1; i >= 0; i--){
        if (this->_tensor_data->_grad_fn[i]){
            this->_tensor_data->_grad_fn[i]->backward(prop_grad, this);
        }
    }
}

template<Scalar T>
CTensor<T> CTensor<T>::clone() {
    CTensor<T> Cloned_CTensor(new DTensor<T>(*_tensor_data));
    return Cloned_CTensor;
}
/* untestee
template<Scalar T>
CTensor<T> zeros(std::vector<size_t> shape) {
    std::vector<T> data(stride(-1,shape),T(0));
    return CTensor<T>(data, shape);
}

template<Scalar T>
CTensor<T> ones(std::vector<size_t> shape) {
    std::vector<T> data(stride(-1,shape),T(1));
    return CTensor<T>(data, shape);
}

template<Scalar T>
CTensor<T> random(std::vector<size_t> shape, T min, T max) {
    return CTensor<T>(randomVector<T>(stride(-1,shape),min,max),shape);
}
    
template<typename T,Container U>
CTensor<T> Tensor(U data) {
    return CTensor<T>(data);
}

template<typename T, Scalar U>
CTensor<T> Tensor(U data) {
    return CTensor<T>(data);
}

template<typename T>
CTensor<T> Tensor(std::vector<T> data, std::vector<size_t> shape) {
    return CTensor<T>(data,shape);
}
*/

} //namespace

#endif