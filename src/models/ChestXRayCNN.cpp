#include "ChestXRayCNN.h"

ChestXRayCNN::ChestXRayCNN()
{
    features = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2), // 112x112

        torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2), // 56x56

        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2), // 28x28

        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2) // 14x14
    );

    // 128 x 14 x 14 = 25088
    fc1 = torch::nn::Linear(128 * 14 * 14, 256);
    fc2 = torch::nn::Linear(256, 2);  // Binary classification (logits for 2 classes)

    register_module("features", features);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor ChestXRayCNN::forward(torch::Tensor x)
{
    x = features->forward(x);
    x = x.view({x.size(0), -1}); // Flatten
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.5, is_training());
    embedding_ = x.clone();
    x = fc2->forward(x);
    return torch::log_softmax(x, /*dim=*/1);
}
