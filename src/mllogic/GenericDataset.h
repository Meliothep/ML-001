#ifndef GENERICDATASET_H
#define GENERICDATASET_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <optional>
#include "ModelConcepts.h"

template <typename Self>
class GenericDataset : public torch::data::Dataset<Self>, public Nameable {
protected:
    std::vector<torch::Tensor> labels_;

    virtual void read_data(const std::string loc, bool isTrain) = 0;

public:
    GenericDataset(const std::string& loc_data, bool isTrain) {
    }

    void initialize(const std::string loc_data, bool isTrain) {
        read_data(loc_data, isTrain);
    }

    torch::data::Example<> get(size_t index) override {
        return {get_sample(index), labels_[index]};
    }

    std::optional<size_t> size() const override {
        return labels_.size();  // Only need label count
    }

    // To be implemented by subclass depending on data source
    virtual torch::Tensor get_sample(size_t index) = 0;
};

#endif // GENERICDATASET_H
