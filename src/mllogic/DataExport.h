//
// Created by bervo on 05/05/2025.
//

#ifndef DATAEXPORT_H
#define DATAEXPORT_H

#include <iostream>
#include "Logger.h"
#include <string>
#include <vector>

struct EpochMetrics {
  int epoch;
  float train_loss;
  float test_loss;
  float accuracy;
};

struct PredictionSample {
  int epoch;
  int sample_index;
  int predicted_label;
  int true_label;
  float confidence;
  std::vector<float> embedding; // for projection maps like t-SNE
};

class DataExporter {
protected :
  std::string output_path_;
  std::shared_ptr<Logger> logger_;

public:
  virtual ~DataExporter() = default;

  explicit DataExporter(std::string output_path, std::shared_ptr<Logger> logger) : output_path_(output_path), logger_(logger) {}

  virtual int ExportEpochMetrics(std::string output_file, const std::vector<EpochMetrics>& metrics) = 0;
  virtual int ExportPredictions(std::string output_file, const std::vector<PredictionSample>& samples) = 0;
};
#endif // DATAEXPORT_H
