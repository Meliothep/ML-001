//
// Created by bervo on 06/07/2025.
//

#ifndef RESNETBLOCK_H
#define RESNETBLOCK_H

#include <torch/torch.h>

class ResNetBlockImpl : public torch::nn::Module {
public:
    ResNetBlockImpl(int64_t in_channels, int64_t out_channels, int64_t stride = 1, bool downsample = false)
        : conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                    .stride(stride).padding(1).bias(false))),
          bn1(torch::nn::BatchNorm2d(out_channels)),
          conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                                    .stride(1).padding(1).bias(false))),
          bn2(torch::nn::BatchNorm2d(out_channels))
    {
        if (downsample || stride != 1 || in_channels != out_channels) {
            shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                                    .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels)
            );
            register_module("shortcut", shortcut);
        }

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = shortcut.is_empty() ? x : shortcut->forward(x);
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        x += residual;
        return torch::relu(x);
    }

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential shortcut{nullptr};
};

TORCH_MODULE(ResNetBlock);

#endif //RESNETBLOCK_H