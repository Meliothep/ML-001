#ifndef CMAKE_CHESTXRAYDATASET_H
#define CMAKE_CHESTXRAYDATASET_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include "mllogic/GenericDataset.h"

class ChestXRayDataset : public GenericDataset<ChestXRayDataset>
{
public:
    ChestXRayDataset(const std::string& loc_data, bool isTrain)
        : GenericDataset(loc_data, isTrain) {
        initialize(loc_data, isTrain);
    }

    std::string get_name() const override {
        return "ChestXRay";
    }

    torch::Tensor get_sample(size_t index) override;

protected:
    std::vector<torch::Tensor> data_;

    void read_data(const std::string loc, bool isTrain) override;
};

#endif // CMAKE_CHESTXRAYDATASET_H
