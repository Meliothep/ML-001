#ifndef CMAKE_MODELTRAINER_H
#define CMAKE_MODELTRAINER_H

#include <torch/torch.h>


class ModelTrainer {
private:

public:
    ModelTrainer() {

    };
    virtual ~ModelTrainer() {};
};

template<typename T, typename std::enable_if<std::is_base_of<torch::nn::Module, T>::value>::type* = nullptr>
T Foo(T bar)
{
    return T();
}

// Function to train the model
template <typename DataLoader, typename Model>
void train(
    std::shared_ptr<Model> model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t epoch,
    size_t dataset_size) {
   
    model->train();
    size_t batch_idx = 0;
    float running_loss = 0.0;
    
    // Calculate total number of batches
    size_t total_batches = std::ceil(static_cast<float>(dataset_size) / data_loader.options().batch_size);
   
    for (auto& batch : data_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
       
        optimizer.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nll_loss(output, target);
       
        loss.backward();
        optimizer.step();
       
        running_loss += loss.item<float>();
       
        if (batch_idx % 100 == 0) {
            std::cout << "Train Epoch: " << epoch << " ["
                      << batch_idx * batch.data.size(0) << "/" << dataset_size
                      << " (" << 100. * batch_idx / total_batches << "%)]"
                      << "\tLoss: " << loss.item<float>() << std::endl;
        }
       
        batch_idx++;
    }
   
    std::cout << "Train Epoch: " << epoch << " complete, Avg Loss: "
              << running_loss / batch_idx << std::endl;
}

// Function to test the model
template <typename DataLoader, typename Model>
void test(
    std::shared_ptr<Model> model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
    
    model->eval();
    float test_loss = 0;
    int32_t correct = 0;
    
    torch::NoGradGuard no_grad;
    
    for (const auto& batch : data_loader) {
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        
        auto output = model->forward(data);
        test_loss += torch::nll_loss(output, target, {}, torch::Reduction::Sum).item<float>();
        
        auto pred = output.argmax(1);
        correct += pred.eq(target).sum().template item<int64_t>();
    }
    
    test_loss /= dataset_size;
    
    std::cout << "Test set: Average loss: " << test_loss 
              << ", Accuracy: " << correct << "/" << dataset_size
              << " (" << 100.0 * correct / dataset_size << "%)" << std::endl;
}

#endif