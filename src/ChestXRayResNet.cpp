//
// Created by bervo on 06/07/2025.
//

#include "ChestXRayResNet.h"

ChestXRayResNet::ChestXRayResNet() {
    // Initial layers
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3).bias(false));
    bn1 = torch::nn::BatchNorm2d(64);
    relu = torch::nn::ReLU();
    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

    // ResNet layers
    layer1 = make_layer(64, 64, 2, 1);
    layer2 = make_layer(64, 128, 2, 2);
    layer3 = make_layer(128, 256, 2, 2);
    layer4 = make_layer(256, 512, 2, 2);

    // Global average pooling and classifier
    avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    dropout = torch::nn::Dropout(torch::nn::DropoutOptions(0.5));
    fc = torch::nn::Linear(512, 2);

    // Register all modules
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("relu", relu);
    register_module("maxpool", maxpool);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("avgpool", avgpool);
    register_module("dropout", dropout);
    register_module("fc", fc);
}

torch::nn::Sequential ChestXRayResNet::make_layer(int64_t in_channels, int64_t out_channels, int blocks, int64_t stride) {
    torch::nn::Sequential layer;

    // First block with potential downsampling
    layer->push_back(ResNetBlock(in_channels, out_channels, stride, stride != 1 || in_channels != out_channels));

    // Remaining blocks
    for (int i = 1; i < blocks; ++i) {
        layer->push_back(ResNetBlock(out_channels, out_channels, 1, false));
    }

    return layer;
}

torch::Tensor ChestXRayResNet::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu->forward(x);
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});  // Flatten
    x = dropout->forward(x);      // Add dropout before FC layer
    embedding_ = x.clone();       // Save embedding (after dropout)
    x = fc->forward(x);

    return torch::log_softmax(x, /*dim=*/1);
}