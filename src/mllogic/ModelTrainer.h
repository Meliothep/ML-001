#ifndef CMAKE_MODELTRAINER_H
#define CMAKE_MODELTRAINER_H

#include <torch/torch.h>
#include "GenericDataset.h"
#include "Logger.h"
#include "DataExport.h"
#include <vector>

template<typename Model, typename Dataset>
class ModelTrainer {
public:
    ModelTrainer(std::string data_dir,
                 std::shared_ptr<DataExporter> data_exporter,
                 std::shared_ptr<Logger> logger);

    virtual ~ModelTrainer() = default;

    /// Runs full training/testing loop and exports data.
    int start();

    /// Saves model weights to `<model_name>.pt`
    void save();

private:
    // core state
    std::shared_ptr<Model>           model_;
    std::string                      data_dir_;
    std::shared_ptr<Logger>          logger_;
    std::shared_ptr<DataExporter>    data_exporter_;

    // logging
    std::vector<EpochMetrics>        metrics_;
    std::vector<PredictionSample>    predictions_;

    // train / test routines
    template<typename DataLoader>
    void train(std::shared_ptr<Model> model,
               torch::Device device,
               DataLoader& loader,
               torch::optim::Optimizer& optimizer,
               size_t epoch,
               size_t dataset_size);

    template<typename DataLoader>
    void test(std::shared_ptr<Model> model,
              torch::Device device,
              DataLoader& loader,
              size_t dataset_size,
              size_t epoch,
              float& out_test_loss,
              float& out_accuracy);

    // per-sample logging
    void log_predictions(
                         size_t epoch,
                         const torch::Tensor& preds,
                         const torch::Tensor& targets,
                         const torch::Tensor& probs,
                         const torch::Tensor& embeddings,
                         int& sample_index);
};

#include "ModelTrainer.cpp"  // include template definitions
#endif
