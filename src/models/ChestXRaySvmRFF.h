#ifndef CHESTXRAYSVMRFF_H
#define CHESTXRAYSVMRFF_H

#include <torch/torch.h>
#include "../mllogic/ModelConcepts.h"

class ChestXRaySvmRFF : public torch::nn::Module, public Nameable, public Embeddable {
public:
    ChestXRaySvmRFF();

    torch::Tensor forward(torch::Tensor x);
    std::string get_name() const override;
    torch::Tensor get_embedding() const override;

private:
    torch::Tensor embedding_;

    double gamma_ = 1e-4;
    int64_t input_dim_ = 224 * 224;
    int64_t rff_dim_ = 5000;
    int64_t num_classes_ = 2;

    torch::Tensor W_; // trainable
    torch::Tensor b_; // trainable
    torch::nn::Linear svm_linear_{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

#endif // CHESTXRAYSVMRFF_H
