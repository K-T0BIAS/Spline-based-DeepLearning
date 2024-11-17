#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "splines.hpp"

namespace SplineNetLib {
    
class base_activation{
    public:
        // Virtual method for the forward pass
        virtual std::vector<double> forward(const std::vector<double>& input) const = 0;
        
        // Virtual method for the backward pass (gradient calculation)
        virtual std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& d_output) const = 0;
        
        // Virtual destructor
        virtual ~base_activation() = default;
};

class layer{
    private:
        
        
        unsigned int in_size, out_size, detail; //num input params,num output params, num of points in all layerspecific splines - 2
        
        std::vector<std::vector<spline>> l_splines;
        std::shared_ptr<base_activation> activation;//new
        bool 
        
    public:
        
        double lr=0.1;//learning_rate
        std::vector<double> last_output;
        //init with input size and target output size aswell as detail and maximum inpjt value
        layer(unsigned int _in_size,unsigned int _out_size,unsigned int _detail,double max, base_activation *_activation = nullptr);
        //load from existing layer data
        layer(std::vector<std::vector<std::vector<std::vector<double>>>> points_list,
              std::vector<std::vector<std::vector<std::vector<double>>>> params_list,
              base_activation *_activation = nullptr);
        
        //call interpolation on all l_splines
        void interpolate_splines();
        //calculate n outputs based one m inputs
        std::vector<double> forward(std::vector<double> x,bool normalize);
        //calculate gradient with respect to individual spline than sum up for prev layer->backward (=>d_y or if is last layer d_y=loss gradient)
        std::vector<double> backward(std::vector<double> x,std::vector<double> d_y,std::vector<double> y);
};

}//namespace

#endif