#include "ChestXRayMLPSVM.h"

ChestXRayMLPSVM::ChestXRayMLPSVM() {
    const int64_t input_features = 224 * 224;
    const int64_t hidden_dim = 256;

    mlp_ = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(input_features, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, hidden_dim / 2),
        torch::nn::ReLU()
    ));

    svm_linear_ = register_module("svm_linear", torch::nn::Linear(hidden_dim / 2, 1));  // binary output
}

std::string ChestXRayMLPSVM::get_name() const {
    return "ChestXRayMLPSVM";
}

torch::Tensor ChestXRayMLPSVM::get_embedding() const {
    return embedding_;
}

torch::Tensor ChestXRayMLPSVM::forward(torch::Tensor x) {
    torch::Device device = svm_linear_->parameters()[0].device();
    x = x.to(device);

    torch::Tensor flattened_x = x.view({x.size(0), -1}).to(torch::kFloat32);

    torch::Tensor features = mlp_->forward(flattened_x);

    embedding_ = features.clone();

    torch::Tensor output = svm_linear_->forward(features).squeeze(1);  // (batch_size,) logits

    return output;
}
