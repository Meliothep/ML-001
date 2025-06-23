#include "MNISTDataset.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include "mllogic/utils.hpp"

const std::string trainPrefix = "train";
const std::string testPrefix = "t10k";

void MNISTDataset::read_data(const std::string loc, bool isTrain)
{
    std::cout << "MNISTDataset::read_data" << std::endl;

    std::string path = loc + "/" + (isTrain ? trainPrefix : testPrefix) + "-images.idx3-ubyte";
    std::ifstream images(path, std::ios::binary);
    TORCH_CHECK(images, "Error opening images file at ", path);

    // Read magic number (should be 2051 for MNIST images)
    uint32_t magicNumber = read_int32(images);
    TORCH_CHECK(magicNumber == 2051, "Incorrect magic number in image file: ", magicNumber);

    // Read image dimensions
    uint32_t numImages = read_int32(images);
    uint32_t numRows   = read_int32(images);
    uint32_t numCols   = read_int32(images);

    std::cout << "Reading " << numImages << " images of size " << numRows << "x" << numCols << std::endl;

    // Read all raw data
    std::vector<uint8_t> buffer(numImages * numRows * numCols);
    images.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    // Clear and reserve space in data_ vector
    data_.clear();
    data_.reserve(numImages);

    // Fill data_ with 1x28x28 tensors
    size_t imageSize = numRows * numCols;
    for (size_t i = 0; i < numImages; ++i) {
        auto img_tensor = torch::from_blob(
            buffer.data() + i * imageSize,
            {1, static_cast<long>(numRows), static_cast<long>(numCols)},
            torch::kUInt8
        ).clone().to(torch::kFloat32).div_(255);  // clone to own memory
        //img_tensor = img_tensor .sub_(0.5).div_(0.5);
        data_.push_back(img_tensor);
    }

    read_labels(loc, isTrain);
}

void MNISTDataset::read_labels(const std::string loc, bool isTrain)
{
    std::cout << "MNISTDataset::read_labels" << std::endl;
    std::string path = loc + "/" + (isTrain ? trainPrefix : testPrefix) + "-labels.idx1-ubyte";

    std::ifstream targets(path, std::ios::binary);
    TORCH_CHECK(targets, "Error opening targets file at ", path);

    // Read magic number (should be 2049 for MNIST labels)
    uint32_t magicNumber = read_int32(targets);
    TORCH_CHECK(magicNumber == 2049, "Incorrect magic number in label file: ", magicNumber);

    // Read number of labels
    uint32_t numLabels = read_int32(targets);

    std::cout << "Reading " << numLabels << " labels" << std::endl;

    // Read raw labels into a buffer
    std::vector<uint8_t> buffer(numLabels);
    targets.read(reinterpret_cast<char*>(buffer.data()), numLabels);

    // Store each label as a scalar tensor in the vector
    labels_.clear();
    labels_.reserve(numLabels);

    for (uint8_t val : buffer) {
        labels_.push_back(torch::tensor(static_cast<int64_t>(val), torch::kInt64));
    }
}

torch::Tensor MNISTDataset::get_sample(size_t index)
{
    return data_[index];
}