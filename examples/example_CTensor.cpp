#include "../include/SplineNetLib/layers.hpp"

using namespace SplineNetLib;

int main() {
    
    //this will create a CTensor that holds a data vector and shape vector, all other member variables are uninitialized
    auto a = CTensor({1,1,1,2,2,2},{2,3});
    
    std::cout<<"created CTensor a with data : "<<vectorToString(a.data())<<" and shape : "<<vectorToString(a.shape())<<"\n";
    
    std::vector<std::vector<int>> data_b({{1,1,1},{2,2,2}});
    
    auto b = CTensor<int>(data_b);
    
    std::cout<<"created CTensor b with data : "<<vectorToString(b.data())<<" and shape : "<<vectorToString(b.shape())<<"\n";
    
    auto c = a + b;
    
    std::cout<<"created CTensor c by adding a + b. Data : "<<vectorToString(c.data())<<" and shape : "<<vectorToString(c.shape())<<"\n";
    
    return 0;
}