#include "ChestXRaySvmRFF.h"
#include <cmath>

ChestXRaySvmRFF::ChestXRaySvmRFF() {
    // Create trainable parameters and register them
    W_ = register_parameter(
        "W_", torch::randn({input_dim_, rff_dim_}) * std::sqrt(2.0 * gamma_));
    b_ = register_parameter(
        "b_", torch::rand({rff_dim_}) * 2 * M_PI);

    W_ = W_.to(torch::kFloat32);
    b_ = b_.to(torch::kFloat32);

    dropout = torch::nn::Dropout(torch::nn::DropoutOptions(0.2));

    register_module("dropout", dropout);
    svm_linear_ = register_module("svm_linear", torch::nn::Linear(rff_dim_, num_classes_));
}

std::string ChestXRaySvmRFF::get_name() const {
    return "ChestXRaySvmRFF";
}

torch::Tensor ChestXRaySvmRFF::get_embedding() const {
    return embedding_;
}

torch::Tensor ChestXRaySvmRFF::forward(torch::Tensor x) {
    x = x.to(torch::kFloat32);
    x = x.view({x.size(0), -1});  // flatten (B, 224*224)

    // RFF projection
    torch::Tensor projection = torch::matmul(x, W_) + b_;
    torch::Tensor z = std::sqrt(2.0 / rff_dim_) * torch::cos(projection);
    z = dropout->forward(z);
    embedding_ = z.clone();

    return svm_linear_->forward(z);
}
