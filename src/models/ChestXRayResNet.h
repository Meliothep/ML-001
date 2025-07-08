//
// Created by bervo on 06/07/2025.
//

#ifndef CHESTXRAYRESNET_H
#define CHESTXRAYRESNET_H

#include "ResNetBlock.h"
#include "ModelConcepts.h"
#include <torch/torch.h>

class ChestXRayResNet : public torch::nn::Module, public Nameable, public Embeddable {
public:
    ChestXRayResNet();
    torch::Tensor forward(torch::Tensor x);

    std::string get_name() const override {
        return "ChestXRayResNet";
    }

    torch::Tensor get_embedding() const override {
        return embedding_;
    }

private:
    // Initial layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};

    // ResNet layers
    torch::nn::Sequential layer1{nullptr};
    torch::nn::Sequential layer2{nullptr};
    torch::nn::Sequential layer3{nullptr};
    torch::nn::Sequential layer4{nullptr};

    // Final layers
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear fc{nullptr};
    
    // Embedding storage
    mutable torch::Tensor embedding_;

    torch::nn::Sequential make_layer(int64_t in_channels, int64_t out_channels, int blocks, int64_t stride = 1);
};

#endif //CHESTXRAYRESNET_H