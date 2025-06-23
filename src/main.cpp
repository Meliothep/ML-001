#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <variant>

#include <torch/torch.h>

#include "ChestXRayDataset.h"
#include "ChestXRayCNN.h"
#include "DataExport.h"
#include "ModelTrainer.h"
#include "Logger.h"
#include "ParquetExport.h"
#include <opencv2/opencv.hpp>

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

    std::shared_ptr<ModelTrainer<ChestXRayCNN,ChestXRayDataset>> trainer(
        new ModelTrainer<ChestXRayCNN,ChestXRayDataset>(argv[1], parquet_data_exporter ,logger));

    trainer->start();
    trainer->save();

    return 0;
}