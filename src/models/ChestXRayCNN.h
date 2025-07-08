#ifndef CMAKE_CHESTXRAYCNN_H
#define CMAKE_CHESTXRAYCNN_H

#include "ModelConcepts.h"
#include <torch/torch.h>

class ChestXRayCNN final : public torch::nn::Module, public Nameable, public Embeddable
{
public:
    ChestXRayCNN();
    torch::Tensor forward(torch::Tensor x);

    std::string get_name() const {
        return "ChestXRayCNN";
    }

    torch::Tensor get_embedding() const {
        return embedding_;
    }

private:
    torch::nn::Sequential features;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::Tensor embedding_;
};

#endif // CMAKE_CHESTXRAYCNN_H
