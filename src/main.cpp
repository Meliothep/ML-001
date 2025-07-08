#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <variant>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "DataExport.h"
#include "ModelTrainer.h"
#include "Logger.h"
#include "ParquetExport.h"

#include "ChestXRayDataset.h"

#include "models/ChestXRayCNN.h"
#include "models/ChestXRayResNet.h"
#include "models/ChestXRaySVM.h"
#include "models/ChestXRaySvmRFF.h"
#include "models/ChestXRayMLPSVM.h"

using namespace cv;


int main(int argc, char* argv[]) {

    std::shared_ptr<Logger> logger(new Logger());
    logger->AddLoggerStrat(new ConsoleLoggerStrategy(LogLevel::INFO));

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_directory>" << std::endl;
        return 1;
    }

    std::shared_ptr<ParquetDataExporter> parquet_data_exporter(
        new ParquetDataExporter(argv[2], logger));

    std::shared_ptr<ModelTrainer<ChestXRayMLPSVM,ChestXRayDataset>> trainer(
        new ModelTrainer<ChestXRayMLPSVM,ChestXRayDataset>(argv[1], parquet_data_exporter ,logger));

    Hypermeters hypermeters = {
        32,      // batch_size
        10,      // num_epochs
        0.0001,  // learning_rate
        0.9,     // momentum
        1e-4,     // weight_decay
        5,       // patience (increase from 3 to 5)
        0.01      // min_delta
    };

    trainer->start(hypermeters);
    trainer->save();

    return 0;
}