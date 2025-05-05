#include "MNISTDataset.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include "mllogic/utils.hpp"

namespace {
    // Put these in anonymous namespace to limit their scope to this file
    const std::string trainPrefix = "train";
    const std::string testPrefix = "t10k";
}

MNISTDataset::MNISTDataset(const std::string &loc_data, bool isTrain)
    : GenericDataset(loc_data, isTrain)
{
    // Call initialize after construction
    initialize(loc_data, isTrain);
}

torch::Tensor MNISTDataset::read_data(const std::string &loc, bool isTrain)
{
    // Load images
    std::string path = loc + (isTrain ? trainPrefix : testPrefix) + "-images.idx3-ubyte";
    std::ifstream images(path, std::ios::binary);
    TORCH_CHECK(images, "Error opening images file at ", path);

    // Read magic number (should be 2051 for MNIST images)
    uint32_t magicNumber = read_int32(images);
    TORCH_CHECK(magicNumber == 2051, "Incorrect magic number in image file: ", magicNumber);

    // Read dimensions
    uint32_t numImages = read_int32(images);
    uint32_t numRows = read_int32(images);
    uint32_t numCols = read_int32(images);

    std::cout << "Reading " << numImages << " images of size " << numRows << "x" << numCols << std::endl;

    // This converts images to tensors
    // Allocate an empty tensor of size of image (count, channels, height, width)
    auto tensor = torch::empty({numImages, 1, numRows, numCols}, torch::kByte);

    // Read image and convert to tensor
    images.read(reinterpret_cast<char *>(tensor.data_ptr()), tensor.numel());

    // Normalize the image from 0 to 255 to 0 to 1
    return tensor.to(torch::kFloat32).div_(255);
}

torch::Tensor MNISTDataset::read_labels(const std::string &loc, bool isTrain)
{
    std::string path = loc + (isTrain ? trainPrefix : testPrefix) + "-labels.idx1-ubyte";
    // Read the labels
    std::ifstream targets(path, std::ios::binary);
    TORCH_CHECK(targets, "Error opening targets file at ", path);

    // Read magic number (should be 2049 for MNIST labels)
    uint32_t magicNumber = read_int32(targets);
    TORCH_CHECK(magicNumber == 2049, "Incorrect magic number in label file: ", magicNumber);

    // Read number of labels
    uint32_t numLabels = read_int32(targets);

    std::cout << "Reading " << numLabels << " labels" << std::endl;

    // Allocate an empty tensor of size of number of labels
    auto tensor = torch::empty(numLabels, torch::kByte);

    // Convert to tensor
    targets.read(reinterpret_cast<char *>(tensor.data_ptr()), numLabels);
    return tensor.to(torch::kInt64);
}