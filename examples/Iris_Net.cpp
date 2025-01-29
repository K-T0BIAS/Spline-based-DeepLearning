// Copyright (c) <2024>, <Tobias Karusseit>
// 
// This file is part of the PySplineNetLib project, which is licensed under the 
// Mozilla Public License, Version 2.0 (MPL-2.0).
// 
// SPDX-License-Identifier: MPL-2.0
// For the full text of the licenses, see:
// - Mozilla Public License 2.0: https://opensource.org/licenses/MPL-2.0

//This example is based on the iris dataset by R.A Fisher 1936
//please download the csv here -> https://archive.ics.uci.edu/dataset/53/iris 
//move the .csv file into the same directory or manually change the path in line 27

//you can run this something like This ↓ (first check where lib was installed, usually at usr/local/)
//g++ -Ipath/to/library/include -Lpath/to/library/lib Iris_Net.cpp -o Iris_Net -lSplineLibNet

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "SplineNetLib/SplineNet.hpp"

std::vector<std::vector<std::vector<double>>> load_data() {
    
    std::vector<std::vector<std::vector<double>>> data; //{{{data x 4},{label like [0,0,1]}},{}}
    std::ifstream file("IRIS.csv"); // Open the file 
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file. please verifybthe file path in line 27" << std::endl;
        
    }

    std::string line;
    while (std::getline(file, line)) { // Read each line from the file
    
        std::vector<double> values; // Store the split values
        std::vector<double> label;
        bool valid = true;
        unsigned int data_count = 0;
        std::stringstream ss(line);     // Create a stringstream from the line
        std::string value;
        
        while (std::getline(ss, value, ',')) { // Split by comma
            if (data_count < 4){//input data
                try {
                    values.push_back(std::stod(value));
                } catch (const std::invalid_argument &e) {
                    std::cerr << "exception " << e.what() << "\n";
                    valid = false;
                    break;
                }
            }
            else {//labels
                if (!value.empty() && value.back() == '\n' || value.back() =='\r') {
                    value.erase(value.length() - 1);  // Remove the newline character
                }
                if (value == "Iris-setosa") {
                    label = {1.0,0.0,0.0};
                } else if (value == "Iris-versicolor") {
                    label = {0.0,1.0,0.0};
                } else if (value == "Iris-virginica") {
                    label = {0.0,0.0,1.0};
                } else {
                    
                    std::cerr << "value is: [" << value << "]\n";
                    valid = false;
                    break;
                }
            }
            data_count++;
        }
        
        if (valid && values.size() == 4 && label.size() == 3) {
            data.push_back({values,label});
        }
/*
        // Output the split values
        std::cout << "Line: ";
        for (const auto& val : values) {
            std::cout << "[" << val << "] ";
        }
        std::cout << std::endl;
*/
    }

    file.close(); // Close the file
    return data;
}

std::vector<std::vector<std::vector<std::vector<double>>>> batching (std::vector<std::vector<std::vector<double>>> data, int batch_size) {
    
    std::vector<std::vector<std::vector<std::vector<double>>>> batched_data; //{{{{data},{label}},{{...},{...}},...},{batch}}
    std::vector<std::vector<std::vector<double>>> batch; //{{{data},{label}},{{...},{...}},...}
    unsigned int counter = 0;
    
    //for single data point in full data
    for (std::vector<std::vector<double>> single : data) {
        //add to batch
        batch.push_back(single);
        counter++;
        //if max batch size was reached
        if (counter == batch_size) {
            batched_data.push_back(batch); //add batch to new batched data
            counter = 0;
        }
    }
    //if sample % batch_size !=0
    if (!batch.empty()) {
        batched_data.push_back(batch);
    }
    
    return batched_data;
}

template<typename T = std::vector<double>>
class LOSS {
public:

    T loss;
    T loss_grad;

    LOSS(const T &_loss, const T &_loss_grad) : loss(_loss), loss_grad(_loss_grad) {}
    
    const T& item(){
        return loss;
    }
    
    const T& backward(){
        return loss_grad;
    }
    

};

class MSE_loss {

public:
    
    MSE_loss(){}
    
    LOSS<> operator()(const std::vector<double> &x, const std::vector<double> &y){
        std::vector<double> loss_grad(y.size());
        std::vector<double> loss(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            double diff =  y[i] - x[i];
            loss[i] = diff * diff;
            loss_grad[i] = 2 * diff;
        }
        return LOSS<> (loss, loss_grad);
    }
    
    LOSS<std::vector<std::vector<double>>> operator()(const std::vector<std::vector<double>> &x, const std::vector<std::vector<double>> &y){
        std::vector<std::vector<double>> loss (x.size(),std::vector<double>(x[0].size(),0.0));
        std::vector<std::vector<double>> loss_grad (x.size(),std::vector<double>(x[0].size(),0.0));
        for (size_t i = 0; i < x.size(); i++) {
            LOSS<> temp_loss = (*this)(x[i],y[i]);
            loss[i] = temp_loss.item();
            loss_grad[i] = temp_loss.backward();
        }
        return LOSS<std::vector<std::vector<double>>> (loss, loss_grad);
    }
    
    
};

class Sigmoid {
    
private: 
    
    std::vector<double> sample_out;
    std::vector<std::vector<double>> batch_out;
    
    
    double sig(const double &x) {
        return 1.0 / (1.0 + exp(-1.0 * x));
    }
public:
    
    Sigmoid() {}
    
    std::vector<double> forward(const std::vector<double> &x) {
        std::vector<double> out (x.size(),0.0);
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = sig(x[i]);
        }
        sample_out = out;
        return out;
    }
    
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>> &x) {
        std::vector<std::vector<double>> out (x.size(), std::vector<double>(x[0].size(),0.0));
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = forward(x[i]);
        }
        batch_out = out;
        return out;
    }

    
    std::vector<double> backward(const std::vector<double> &d_y) {
        std::vector<double> out (sample_out.size(),0.0);
        for (size_t i = 0; i < sample_out.size(); i++) {
            double temp = sample_out[i];
            out[i] = d_y[i] * temp * (1.0 - temp);
        }
        return out;
    }
    
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>> &d_y) {
        std::vector<std::vector<double>> out;
        for (size_t i = 0; i < batch_out.size(); i++) {
            sample_out = batch_out[i];
            auto temp = backward(d_y[i]);
            out.push_back(temp); 
        }
        return out;
    }
};


int main() {
    int batch_size = 20; //num of samples per batch (must be >1)
    int num_epochs = 50; //num of training cycles (epochs)
    double lr = 0.001;   //learning rate, determines the strength of the applied grads. high lr might couse no convergance low lr might need more epochs
    
    auto dataset = batching(load_data(),batch_size); //load the data in batched format
    
    SplineNetLib::layer Layer_A = SplineNetLib::layer(4,3,9,10.0); //Create a new layer with x points: 0,1,2,3,4,5,6,7,8,9,10 
                                                                   //takes vectors with 4 entries and returns vector with 3 entries
    Layer_A.lr = lr; //set the layer lr to the specified lr
    
    //not actually necessary but good practice
    Layer_A.interpolate_splines(); //init layer parameters

    
    MSE_loss loss_func = MSE_loss(); //define mse loss functiom
    Sigmoid activation = Sigmoid(); //define activation func (IMPORTANT this will ensure all layer outputs are between 0 and 1,
                                    //for multi layer nets this is required. otherwise input data will exit the spline ranges)
    
    for (int epoch = 0; epoch < num_epochs; epoch++) { //loop over epochs
        for (const std::vector<std::vector<std::vector<double>>> &batch : dataset) { //loop over data and select new batch every time
            std::vector<std::vector<double>> X,y; //define input and target vectors
            for (const std::vector<std::vector<double>> &sample : batch) { //assign inputs and targets
                X.push_back(sample[0]);
                y.push_back(sample[1]);
            }
            //fwd
            auto x = Layer_A.forward(X,false); //forward pass with auto normalizing of (since activation will do this (this is preffered))
            x = activation.forward(x);         //apply the activation
            auto loss = loss_func(x,y);        //get the loss 
            
            std::cout << "loss = " << loss.item()[0][0] <<" "<< loss.item()[1][0]<< "\n";
            //backward pass through loss, activation amd layer
            auto grad = loss.backward();
            grad = activation.backward(grad);
            auto g = Layer_A.backward(X,grad);
            
        }
    }
    
    std::vector<std::vector<SplineNetLib::spline>> splines = Layer_A.get_splines(); //get a vector of all splines in the layer 
    //print all the spline points 
    for (size_t i = 0; i < splines.size(); i++) {
        for (size_t j = 0; j < splines[0].size(); j++) {
            auto points = splines[i][j].get_points();
            std::cout << "points:\n";
            for (size_t a = 0; a < points.size(); a++) {
                std::cout/*<< "x:"*/<<points[a][0]/*<<" y:"*/<<" "<<points[a][1]<<"\n";
            }
        }
    }
    //some tests taken from the .csv (highest value in the test should match the 1 in the target (correct))
    //!!!note that this is a very small test and the accuracy would have to be calculated over more samples
    std::vector<double> test = {5.9,3.0,5.1,1.8};
    std::vector<double> label = {0.0,0.0,1.0};
    auto t = Layer_A.forward(test,false);
    t = activation.forward(t);
    std::cout << "test: 1"<< t[0] << " "<<t[1] << " "<< t[2] <<"\n";
    std::cout << "correct:"<< label[0]<< label[1] << label[2]<<"\n";
    
    test = {5.8,4.0,1.2,0.2};
    label = {1.0,0.0,0.0};
    t = Layer_A.forward(test,false);
    t = activation.forward(t);
    std::cout << "test: 2"<< t[0] << " "<<t[1] << " "<< t[2] <<"\n";
    std::cout << "correct:"<< label[0]<< label[1] << label[2]<<"\n";
    return 0;
}