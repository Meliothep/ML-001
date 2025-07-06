#ifndef CMAKE_MODELTRAINER_H
#define CMAKE_MODELTRAINER_H

#include <torch/torch.h>
#include "GenericDataset.h"
#include "Logger.h"
#include "DataExport.h"
#include <vector>

struct Hypermeters {
    int64_t batch_size = 64;
    int64_t epochs = 10;
    double  lr = 0.001;
    double  mom = 0.9;
    double weight_decay = 0;
    int patience = 3;
    double min_delta = 0.001;
};

template<typename Model, typename Dataset>
class ModelTrainer {
public:
    ModelTrainer(std::string data_dir,
                 std::shared_ptr<DataExporter> data_exporter,
                 std::shared_ptr<Logger> logger);

    virtual ~ModelTrainer() = default;

    /// Runs full training/testing loop and exports data.
    int start(Hypermeters hypermeters);

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
