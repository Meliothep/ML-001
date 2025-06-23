#include "ChestXRayDataset.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void ChestXRayDataset::read_data(const std::string loc, bool isTrain)
{
    std::cout << "ChestXRayDataset::read_data from " << loc << std::endl;

    std::string train_or_test = isTrain ? "train" : "test";
    std::string base_path = loc + "/" + train_or_test;

    data_.clear();
    labels_.clear();

    for (const auto& entry : fs::recursive_directory_iterator(base_path)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();

            // Use OpenCV to load image
            cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Warning: could not read image: " << path << std::endl;
                continue;
            }


            if (isTrain) {
                // Horizontal flip with 50% chance
                if (rand() % 2 == 0)
                    cv::flip(img, img, 1);

                // Random rotation ±10 degrees
                double angle = ((rand() % 21) - 10);  // [-10, +10]
                cv::Point2f center(img.cols / 2.0F, img.rows / 2.0F);
                cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
                cv::warpAffine(img, img, rot_mat, img.size());

                // Optional: slight brightness jitter
                img.convertTo(img, -1, 1 + ((rand() % 21 - 10) / 100.0), rand() % 11 - 5);
            }

            // Resize to 224x224
            cv::resize(img, img, cv::Size(224, 224));

            // Convert to float tensor
            auto tensor = torch::from_blob(
                img.data,
                {1, img.rows, img.cols},
                torch::kUInt8
            ).clone().to(torch::kFloat32).div_(255); // Normalize to [0,1]

            // trying another normalization
            //tensor = tensor.sub_(0.5).div_(0.5);

            data_.push_back(tensor);

            // Label: 0 = NORMAL, 1 = PNEUMONIA
            std::string parent_dir = entry.path().parent_path().filename().string();
            int64_t label = (parent_dir == "NORMAL") ? 0 : 1;
            labels_.push_back(torch::tensor(label, torch::kInt64));
        }
    }

    std::cout << "Loaded " << data_.size() << " images with labels." << std::endl;
}

torch::Tensor ChestXRayDataset::get_sample(size_t index)
{
    return data_[index];
}
