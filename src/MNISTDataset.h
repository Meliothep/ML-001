#ifndef CMAKE_MNISTDATASET_H
#define CMAKE_MNISTDATASET_H

#include <string>
#include <vector>
#include <iostream>
#include <torch/torch.h>
#include "mllogic/GenericDataset.h"

class MNISTDataset : public GenericDataset<MNISTDataset>
{
public:
    MNISTDataset(const std::string loc_data, bool isTrain): GenericDataset(loc_data, isTrain) {
        initialize(loc_data, isTrain);
    };

    std::string get_name() const override{
        return "MNIST";
    }

    torch::Tensor get_sample(size_t index) override;

protected:
    std::vector<torch::Tensor> data_;

    void read_data(const std::string loc, bool isTrain) override;
    void read_labels(const std::string loc, bool isTrain);
};

#endif // CMAKE_MNISTDATASET_H