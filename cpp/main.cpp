#include <iostream>
#include <iomanip>
#include <chrono>

#include "data.hpp"
#include "model.hpp"

int main() {
    try {

        auto total_start = std::chrono::high_resolution_clock::now();

        std::cout << "Loading MNIST dataset...\n";
        auto train_data = load_mnist_dataset(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte"
        );
        auto test_data = load_mnist_dataset(
            "../data/MNIST/raw/t10k-images-idx3-ubyte",
            "../data/MNIST/raw/t10k-labels-idx1-ubyte"
        );
        
        std::cout << "Loaded " << train_data.size() << " training samples\n";
        std::cout << "Loaded " << test_data.size() << " test samples\n\n";

        MLPNet model(128);
        
        const int num_epochs = 5;
        const int batch_size = 64;
        const float learning_rate = 0.01f;
        
        DataLoader train_loader(train_data, batch_size, true);
        DataLoader test_loader(test_data, batch_size, false);
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {

            auto epoch_start = std::chrono::high_resolution_clock::now();
            std::cout << "=== Epoch " << (epoch + 1) << "/" << num_epochs << " ===\n";
            
            float epoch_loss = 0.0f;
            int batch_count = 0;
            
            train_loader.reset();
            while (train_loader.has_next()) {
                auto batch = train_loader.next_batch();
                float loss = model.train_batch(batch, learning_rate);
                epoch_loss += loss;
                batch_count++;
                
                if (batch_count % 100 == 0) {
                    std::cout << "  Batch " << batch_count << "/" << train_loader.num_batches()
                              << " - Loss: " << std::fixed << std::setprecision(4) << loss << "\n";
                }
            }
            
            float avg_loss = epoch_loss / batch_count;
            
            float test_acc = 0.0f;
            int test_batches = 0;
            test_loader.reset();
            while (test_loader.has_next()) {
                auto batch = test_loader.next_batch();
                test_acc += model.compute_accuracy(batch);
                test_batches++;
            }
            test_acc /= test_batches;
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            std::cout << "Epoch " << (epoch + 1) << " - Avg Loss: " << std::fixed << std::setprecision(4) 
                      << avg_loss << " - Test Accuracy: " << std::setprecision(2) 
                      << (test_acc * 100) << "% - Duration: " << epoch_duration.count() / 1000.0 << " seconds\n\n";
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
        std::cout << "Training complete! Total duration: " << total_duration.count() / 1000.0 << " seconds\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

//g++ -std=c++20 data.cpp model.cpp main.cpp -o mnist_train