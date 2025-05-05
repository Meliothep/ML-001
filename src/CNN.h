#ifndef CMAKE_CNN_H
#define CMAKE_CNN_H

#include <torch/torch.h>

class Net : public torch::nn::Module
{
public:
    Net();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};

#endif