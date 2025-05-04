#ifndef CMAKE_MNISTDATASET_H
#define CMAKE_MNISTDATASET_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <torch/torch.h>

#include "utils.hpp"

std::string trainPrefix = "train";
std::string testprefix = "t10k";

torch::Tensor read_images(const std::string &loc, bool isTrain)
{
    // Load images
    std::string path = loc + (isTrain ? trainPrefix : testprefix) + "-images.idx3-ubyte";
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

torch::Tensor read_labels(const std::string &loc, bool isTrain)
{
    std::string path = loc + (isTrain ? trainPrefix : testprefix) + "-labels.idx1-ubyte";
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

class MyDataset : public torch::data::Dataset<MyDataset>
{
private:
    torch::Tensor images_, labels_;
public:
    explicit MyDataset(const std::string &loc_data, bool isTrain)
        : images_(read_images(loc_data, isTrain)),
          labels_(read_labels(loc_data, isTrain)) {
          };
    torch::data::Example<> get(size_t index) override;
    std::optional<size_t> size() const override {
        return images_.size(0);
    }
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {images_[index], labels_[index]};
}

#endif
