//
// Created by bervo on 05/05/2025.
//

#ifndef GENERICDATASET_H
#define GENERICDATASET_H

#include <torch/torch.h>


template <typename Self>
class GenericDataset : public torch::data::Dataset<Self>
{
private:
    torch::Tensor data_, labels_;

protected:
    // Pure virtual methods (to be implemented by derived classes)
    virtual torch::Tensor read_data(const std::string &loc, bool isTrain) = 0;
    virtual torch::Tensor read_labels(const std::string &loc, bool isTrain) = 0;

public:

    GenericDataset(const std::string &loc_data, bool isTrain)
        : data_(), labels_()
    {
    }

    void initialize(const std::string &loc_data, bool isTrain)
    {
        data_ = read_data(loc_data, isTrain);
        labels_ = read_labels(loc_data, isTrain);
    }

    torch::data::Example<> get(size_t index) override
    {
        return {data_[index], labels_[index]};
    }

    std::optional<size_t> size() const override {
        return data_.size(0);
    }
};

#endif //GENERICDATASET_H
