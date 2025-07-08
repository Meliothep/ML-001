//
// Created by bervo on 08/07/2025.
//

#ifndef CHESTXRAYSVM_H
#define CHESTXRAYSVM_H

#include <torch/torch.h>
#include "../mllogic/ModelConcepts.h"

class ChestXRaySVM : public torch::nn::Module, public Nameable, public Embeddable {
public:
    ChestXRaySVM();

    torch::Tensor forward(torch::Tensor x);

    std::string get_name() const override;

    torch::Tensor get_embedding() const override;

private:
    torch::nn::Linear svm_linear_{nullptr};

    mutable torch::Tensor embedding_;
};




#endif //CHESTXRAYSVM_H
