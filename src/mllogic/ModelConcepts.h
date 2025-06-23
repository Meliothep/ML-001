//
// Created by bervo on 02/06/2025.
//

#ifndef MODELCONCEPTS_H
#define MODELCONCEPTS_H

#include <string>
#include <torch/torch.h>

struct Nameable {
    virtual ~Nameable() = default;
    virtual std::string get_name() const = 0;
};


struct Embeddable {
    virtual ~Embeddable() = default;
    virtual torch::Tensor get_embedding() const = 0;
};



#endif //MODELCONCEPTS_H
