//
// Created by bervo on 08/07/2025.
//

#include "ChestXRaySVM.h"

ChestXRaySVM::ChestXRaySVM() {
    const int64_t input_features = 224 * 224;
    const int64_t num_classes = 2;

    svm_linear_ = register_module("svm_linear", torch::nn::Linear(input_features, num_classes));
}

std::string ChestXRaySVM::get_name() const {
    return "ChestXRaySVM";
}

torch::Tensor ChestXRaySVM::get_embedding() const {
    return embedding_;
}

torch::Tensor ChestXRaySVM::forward(torch::Tensor x) {
    torch::Device device = svm_linear_->parameters()[0].device();
    x = x.to(device);

    torch::Tensor flattened_x = x.view({x.size(0), -1}).to(torch::kFloat32);

    embedding_ = flattened_x.clone();

    torch::Tensor output = svm_linear_->forward(flattened_x);

    return torch::log_softmax(output, /*dim=*/1);
}
