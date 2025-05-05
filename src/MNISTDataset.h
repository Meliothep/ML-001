#ifndef CMAKE_MNISTDATASET_H
#define CMAKE_MNISTDATASET_H

#include <string>
#include <torch/torch.h>
#include "mllogic/GenericDataset.h"

class MNISTDataset : public GenericDataset<MNISTDataset>
{
public:
    MNISTDataset(const std::string &loc_data, bool isTrain);

protected:
    torch::Tensor read_data(const std::string &loc, bool isTrain) override;
    torch::Tensor read_labels(const std::string &loc, bool isTrain) override;
};

#endif // CMAKE_MNISTDATASET_H