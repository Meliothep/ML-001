#ifndef CMAKE_CNN_H
#define CMAKE_CNN_H

#include "ModelConcepts.h"
#include <torch/torch.h>


class MNISTCNN final : public torch::nn::Module, public Nameable, public Embeddable
{
public:
    MNISTCNN();
    torch::Tensor forward(torch::Tensor x);

    virtual std::string get_name() const {
        return "CNN";
    };

    virtual torch::Tensor get_embedding() const {
        return embedding_;
    }

private:
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Dropout2d conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    torch::Tensor embedding_;
};

#endif