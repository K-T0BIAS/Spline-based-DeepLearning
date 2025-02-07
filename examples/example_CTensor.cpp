#include "../include/SplineNetLib/layers.hpp"

using namespace SplineNetLib;

int main() {
    
    //this will create a CTensor that holds a data vector and shape vector, all other member variables are uninitialized
    auto a = CTensor({1,1,1,2,2,2},{2,3});
    
    std::cout<<"created CTensor a with data : "<<vectorToString(a.data())<<" and shape : "<<vectorToString(a.shape())<<"\n";
    
    return 0;
}