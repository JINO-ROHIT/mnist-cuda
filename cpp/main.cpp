#include <iostream>
#include "data.hpp"
#include "model.hpp"

int main() {
    try {
        auto train_data = load_mnist_dataset(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte"
        );
        
        std::cout << "Loaded " << train_data.size() << " training samples.\n";

        MLPNet model(128);
        
        DataLoader loader(train_data, 64, false); // batch_size = 64, no shuffle for testing
        
        int total_samples = 0;
        int total_correct = 0;
        int batch_count = 0;
        
        while (loader.has_next() && batch_count < 10) {
            auto batch = loader.next_batch();
            float batch_acc = model.compute_accuracy(batch);
            
            int batch_correct = static_cast<int>(batch_acc * batch.size());
            total_correct += batch_correct;
            total_samples += batch.size();
            
            std::cout << "Batch " << batch_count << ": " 
                      << batch_correct << "/" << batch.size() 
                      << " correct (" << (batch_acc * 100) << "%)\n";
            
            batch_count++;
        }
        
        float overall_acc = static_cast<float>(total_correct) / total_samples;
        std::cout << "\nOverall accuracy: " << total_correct << "/" << total_samples 
                  << " (" << (overall_acc * 100) << "%)\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

//g++ -std=c++20 data.cpp model.cpp main.cpp -o mnist_test