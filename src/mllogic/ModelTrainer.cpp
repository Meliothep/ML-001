// ModelTrainer.cpp
#ifndef CMAKE_MODELTRAINER_CPP
#define CMAKE_MODELTRAINER_CPP

#define NOMINMAX

#include "ModelTrainer.h"
#include <cmath>
#include <stdexcept>
#include <limits>

// Example training loop with early stopping
class EarlyStopping {
private:
    int patience;
    double min_delta;
    int counter;
    double best_loss;

public:
    EarlyStopping(int patience, double min_delta)
        : patience(patience), min_delta(min_delta), counter(0), best_loss(std::numeric_limits<double>::max()) {}

    bool should_stop(double val_loss) {
        if (val_loss < best_loss - min_delta) {
            best_loss = val_loss;
            counter = 0;
            return false;
        } else {
            counter++;
            return counter >= patience;
        }
    }
};

// ——————————————————————————————————————————————————————
// Constructor
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
ModelTrainer<Model, Dataset>::ModelTrainer(
    std::string data_dir,
    std::shared_ptr<DataExporter> data_exporter,
    std::shared_ptr<Logger> logger)
  : data_dir_(std::move(data_dir))
  , data_exporter_(std::move(data_exporter))
  , logger_(std::move(logger))
{
    if (!std::is_base_of<torch::nn::Module, Model>::value) {
        throw std::invalid_argument("Model must inherit from torch::nn::Module");
    }
    if (!std::is_base_of<GenericDataset<Dataset>, Dataset>::value) {
        throw std::invalid_argument("Dataset must inherit from GenericDataset");
    }
}

// ——————————————————————————————————————————————————————
// start()
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
int ModelTrainer<Model, Dataset>::start(Hypermeters hypermeters) {
    try {

        // device
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            logger_->Log(LogLevel::INFO, "CUDA available. Using GPU.");
            device = torch::kCUDA;
        }

        // datasets + loaders
        auto train_ds = Dataset(data_dir_, true).map(torch::data::transforms::Stack<>());
        size_t train_size = *train_ds.size();
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_ds), hypermeters.batch_size);

        auto test_ds = Dataset(data_dir_, false).map(torch::data::transforms::Stack<>());
        size_t test_size = *test_ds.size();
        auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_ds), hypermeters.batch_size);

        // model + optimizer
        model_ = std::make_shared<Model>();
        model_->to(device);
        torch::optim::SGD optimizer(
            model_->parameters(),
            torch::optim::SGDOptions(hypermeters.lr).momentum(hypermeters.mom).weight_decay(hypermeters.weight_decay)  // Add weight decay
        );

        EarlyStopping early_stopping(hypermeters.patience, hypermeters.min_delta);

        double best_val_loss = std::numeric_limits<double>::max();
        int best_epoch = 0;

        // epochs
        for (int epoch = 1; epoch <= hypermeters.epochs; ++epoch) {
            logger_->Log(LogLevel::INFO, "=== Epoch {} ===", epoch);

            // train
            train(model_, device, *train_loader, optimizer, epoch, train_size);

            // test
            float test_loss = 0, accuracy = 0;
            test(model_, device , *test_loader, test_size, epoch, test_loss, accuracy);

            // record metrics
            metrics_.push_back({epoch,
                                /*train_loss=*/0.0f, // you can store and return it if desired
                                test_loss,
                                accuracy});

            if (early_stopping.should_stop(test_loss)) {
                logger_->Log(LogLevel::INFO, "Early stopping at epoch {}", epoch);
                break;
            }

            if (test_loss < best_val_loss) {
                best_val_loss = test_loss;
                best_epoch = epoch;
                torch::save(model_, model_->get_name() + ".pt");

                logger_->Log(LogLevel::INFO, "New best model saved at epoch ",epoch);
                logger_->Log(LogLevel::INFO, "Model saved to {}.pt", model_->get_name());
            }
        }
    }
    catch (const std::exception& e) {
        logger_->Log(LogLevel::EXCEPT, e.what());
        return 1;
    }
    catch (...) {
        logger_->Log(LogLevel::EXCEPT, "Unknown error");
        return 1;
    }
    return 0;
}

// ——————————————————————————————————————————————————————
// save()
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
void ModelTrainer<Model, Dataset>::save() {
    auto dataset_name = Dataset(data_dir_, true).get_name();
    // export
    data_exporter_->ExportEpochMetrics(model_->get_name() + "_" + dataset_name +  "_metrics" + ".parquet", metrics_);
    data_exporter_->ExportPredictions(model_->get_name() + "_" + dataset_name + "_preds" + ".parquet", predictions_);
}

// ——————————————————————————————————————————————————————
// train()
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
template<typename DataLoader>
void ModelTrainer<Model, Dataset>::train(
    std::shared_ptr<Model> model,
    torch::Device device,
    DataLoader& loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t dataset_size)
{
    model->train();
    size_t batch_idx = 0;
    float running_loss = 0.0f;
    size_t total_batches = std::ceil((float)dataset_size / loader.options().batch_size);

    for (auto& batch : loader) {
        auto data   = batch.data.to(device);
        auto target = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss   = torch::nll_loss(output, target);
        loss.backward();
        optimizer.step();

        float loss_v = loss.item<float>();
        running_loss += loss_v;

        if (batch_idx % 100 == 0) {
            logger_->Log(LogLevel::INFO,
                         "Train: Epoch {} [{}/{} ({:.0f}%)]  Loss: {:.4f}",
                         epoch,
                         batch_idx * batch.data.size(0),
                         dataset_size,
                         100. * batch_idx / total_batches,
                         loss_v);
        }
        batch_idx++;
    }

    logger_->Log(LogLevel::INFO,
                 "Epoch {} train complete, avg loss: {:.4f}",
                 epoch,
                 running_loss / batch_idx);
}

// ——————————————————————————————————————————————————————
// test()
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
template<typename DataLoader>
void ModelTrainer<Model, Dataset>::test(
    std::shared_ptr<Model> model,
    torch::Device device,
    DataLoader& loader,
    size_t dataset_size,
    size_t epoch,
    float& out_test_loss,
    float& out_accuracy)
{
    model->eval();
    torch::NoGradGuard no_grad;
    float test_loss = 0.0f;
    int64_t correct = 0;
    int sample_index = 0;

    for (auto& batch : loader) {
        auto data    = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto output = model->forward(data);
        auto embeddings = model->get_embedding().cpu();
        test_loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum)
                         .item<float>();

        auto probs   = torch::softmax(output, 1);
        auto preds   = probs.argmax(1);
        correct     += preds.eq(targets).sum().item<int64_t>();

        // log per-sample predictions
        log_predictions(epoch, preds.cpu(), targets.cpu(), probs, embeddings, sample_index);
    }

    out_test_loss = test_loss / dataset_size;
    out_accuracy  = static_cast<float>(correct) / dataset_size;

    logger_->Log(LogLevel::INFO,
                 "Test set: avg loss: {:.4f}, accuracy: {}/{} ({:.2f}%)",
                 out_test_loss,
                 correct,
                 dataset_size,
                 100. * out_accuracy);
}

// ——————————————————————————————————————————————————————
// log_predictions()
// ——————————————————————————————————————————————————————
template<typename Model, typename Dataset>
void ModelTrainer<Model, Dataset>::log_predictions(
    const size_t epoch,
    const torch::Tensor& preds,
    const torch::Tensor& targets,
    const torch::Tensor& probs,
    const torch::Tensor& embeddings,
    int& sample_index)
{
    for (int i = 0; i < preds.size(0); ++i) {
        PredictionSample s;
        s.epoch = epoch;
        s.sample_index    = sample_index++;
        s.predicted_label = preds[i].item<int>();
        s.true_label      = targets[i].item<int>();

        // confidence from softmax
        s.confidence = probs[i][preds[i].item<int>()].item<float>();

        // embedding
        for (int j = 0; j < embeddings.size(1); ++j) {
            s.embedding.push_back(embeddings[i][j].item<float>());
        }
        predictions_.push_back(std::move(s));
    }
}

#endif  // CMAKE_MODELTRAINER_CPP
