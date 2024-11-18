#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "splines.hpp"

namespace SplineNetLib {
    


class layer{
    private:
        
        
        unsigned int in_size, out_size, detail; //num input params,num output params, num of points in all layerspecific splines - 2
        
        std::vector<std::vector<spline>> l_splines;
        
        
        
    public:
        
        double lr=0.1;//learning_rate
        std::vector<double> last_output;
        //init with input size and target output size aswell as detail and maximum inpjt value
        layer(unsigned int _in_size,unsigned int _out_size,unsigned int _detail,double max);
        //load from existing layer data
        layer(std::vector<std::vector<std::vector<std::vector<double>>>> points_list,
              std::vector<std::vector<std::vector<std::vector<double>>>> params_list
             );
        
        //call interpolation on all l_splines
        void interpolate_splines();
        //calculate n outputs based one m inputs
        std::vector<double> forward(std::vector<double> x,bool normalize);
        //forward with batches
        std::vector<std::vector<double>> forward(const std::vector<std::vector<double> &x, bool normalize);
        //calculate gradient with respect to individual spline than sum up for prev layer->backward (=>d_y or if is last layer d_y=loss gradient)
        std::vector<double> backward(std::vector<double> x,std::vector<double> d_y,std::vector<double> y, bool apply = TRUE);//y might be unused
        //backward pass for batch inputs
        std::vector<double> backward(const std::vector<std::vector<double>> &x,std::vector<double> d_y);
};

}//namespace

#endif