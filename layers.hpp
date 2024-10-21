#ifndef LAYERS_HPP
#define LAYERS_HPP



#include "smartSplines.hpp"

class layer{
    private:
        
        
        unsigned int in_size, out_size, detail; //num input params,num output params, num of points in all layerspecific splines - 2
        
        
        
        layer(unsigned int _in_size,unsigned int _out_size,unsigned int _detail):// added _ to avoid dublicate names 
            in_size(_in_size),out_size(_out_size),detail(_detail){}
        
        std::vector<std::vector<std::unique_ptr<spline>>> l_splines;
        
    public:
        
        double lr=0.1;//learning_rate
        std::vector<double> last_output;

        
        static std::unique_ptr<layer> create(unsigned int _in_size,unsigned int _out_size,unsigned int _detail,double max);
        
        //init with existing val like [out_size][in_size][detail+2][points=2 || params=4]
        static std::unique_ptr<layer> create(std::vector<std::vector<std::vector<std::vector<double>>>> &points_list,
                                             std::vector<std::vector<std::vector<std::vector<double>>>> &params_list);
        
        //call interpolation on all l_splines
        void interpolate_splines();
        //calculate n outputs based one m inputs
        std::vector<double> forward(std::vector<double> x,bool normalize);
        //calculate gradient with respect to individual spline than sum up for prev layer->backward (=>d_y or if is last layer d_y=loss gradient)
        std::vector<double> backward(std::vector<double> x,std::vector<double> d_y,std::vector<double> y);
};

#endif