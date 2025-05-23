#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <variant>

#include <torch/torch.h>

#include "MNISTDataset.h"
#include "CNN.h"
#include "DataExport.h"
#include "ModelTrainer.h"
#include "Logger.h"

int main(int argc, char* argv[]) {

    Logger* logger = new Logger();
    logger->AddLoggerStrat(new ConsoleLoggerStrategy(LogLevel::INFO));

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_directory>" << std::endl;
        return 1;
    }

    arrow::Status status = dataexport::WriteParquetTest("C:/Users/bervo/Documents/PERSO/test.parquet");
    if (!status.ok()) {
        std::cerr << "Error writing Parquet file: " << status.ToString() << std::endl;
        return 1;
    }

    try {
        std::string data_dir = argv[1];

        // Define hyperparameters
        const int64_t batch_size = 64;
        const int64_t epochs = 10;
        const double learning_rate = 0.01;
        const double momentum = 0.5;

        // Check if CUDA is available
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            logger->Log(LogLevel::INFO, "CUDA is available! Training on GPU.");
            device = torch::kCUDA;
        }

        // Create train dataset
        auto train_dataset = MNISTDataset(data_dir, true).map(torch::data::transforms::Stack<>());
        const size_t train_dataset_size = *train_dataset.size();
        std::cout << "Train dataset loaded successfully with " << train_dataset_size << " samples." << std::endl;
        logger->Log(LogLevel::INFO, "Train dataset loaded successfully with {} samples.", train_dataset_size);
        // Create train data loader
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), batch_size);

        // Create test dataset
        auto test_dataset = MNISTDataset(data_dir, false).map(torch::data::transforms::Stack<>());
        const size_t test_dataset_size = *test_dataset.size();
        logger->Log(LogLevel::INFO, "Test dataset loaded successfully with {} samples.", test_dataset_size);

        // Create test data loader
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_dataset), batch_size);

        // Create model
        auto model = std::make_shared<Net>();
        model->to(device);
        logger->Log(LogLevel::INFO, "Model created and moved to {}.", (device == torch::kCUDA ? "CUDA" : "CPU"));

        // Define optimizer
        torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum));

        // Train and test for multiple epochs
        for (size_t epoch = 1; epoch <= epochs; ++epoch) {
            logger->Log(LogLevel::INFO, "----- Epoch {} -----", epoch);
            // Train the model
            train(model, device, *train_loader, optimizer, epoch, train_dataset_size);

            // Test the model
            test(model, device, *test_loader, test_dataset_size);
        }

        // Save the model
        torch::save(model, "mnist_cnn.pt");
        logger->Log(LogLevel::INFO, "Model saved to mnist_cnn.pt");
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}