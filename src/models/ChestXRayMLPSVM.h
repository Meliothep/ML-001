#ifndef CHESTXRAYMLPSVM_H
#define CHESTXRAYMLPSVM_H

#include <torch/torch.h>
#include "../mllogic/ModelConcepts.h"

class ChestXRayMLPSVM : public torch::nn::Module, public Nameable, public Embeddable {
public:
    ChestXRayMLPSVM();

    torch::Tensor forward(torch::Tensor x);

    std::string get_name() const override;

    torch::Tensor get_embedding() const override;

    // Custom hinge loss for SVM training
    torch::Tensor hinge_loss(torch::Tensor outputs, torch::Tensor labels);

private:
    torch::nn::Sequential mlp_{nullptr};  // nonlinear feature extractor
    torch::nn::Linear svm_linear_{nullptr};  // final linear SVM layer

    mutable torch::Tensor embedding_;
};

#endif //CHESTXRAYMLPSVM_H
