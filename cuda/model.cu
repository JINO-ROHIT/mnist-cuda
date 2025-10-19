#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "data.hpp"

//sub-optimal
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void linear_forward_kernel(const float* input, const float* weights, 
                                     const float* bias, float* output, 
                                     int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = bias[idx]; // or 0?
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + idx];
        }
        output[idx] = fmaxf(0.0f, sum); //relu 
    }
}

//TO-DO pretty sub-optimal
__global__ void softmax_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  
        float max_val = input[0];
        for (int i = 1; i < size; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += expf(input[i] - max_val);
        }
        
        for (int i = 0; i < size; i++) {
            input[i] = expf(input[i] - max_val) / sum;
        }
    }
}

__global__ void cross_entropy_loss_kernel(const float* predictions, int label, 
                                         float* loss) {
    *loss = -logf(predictions[label] + 1e-8f);
}

__global__ void backward_output_kernel(const float* output, int label, 
                                      float* d_output, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_classes) {
        d_output[idx] = output[idx] - (idx == label ? 1.0f : 0.0f);
    }
}

__global__ void linear_backward_kernel(const float* d_output, const float* input,
                                      const float* weights, float* d_input,
                                      float* d_weights, float* d_bias,
                                      int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        d_bias[idx] += d_output[idx];
        for (int i = 0; i < input_size; i++) {
            int weight_idx = i * output_size + idx;
            d_weights[weight_idx] += input[i] * d_output[idx];
            if (input[i] > 0) {  // relu derivative
                atomicAdd(&d_input[i], weights[weight_idx] * d_output[idx]);
            }
        }
    }
}

__global__ void update_weights_kernel(float* weights, float* d_weights, 
                                     int weights_size, float* biases, 
                                     float* d_biases, int biases_size, 
                                     float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weights_size) {
        weights[idx] -= learning_rate * d_weights[idx];
        d_weights[idx] = 0.0f;
    }
    
    if (idx < biases_size && biases != nullptr && d_biases != nullptr) {
        biases[idx] -= learning_rate * d_biases[idx];
        d_biases[idx] = 0.0f;
    }
}

class SimpleMNISTNet {
private:
    static const int input_size = 784;  // 28 x 28
    static const int hidden1_size = 128;
    static const int hidden2_size = 64;
    static const int output_size = 10;

    float* d_w1, *d_b1;
    float* d_w2, *d_b2;
    float* d_w3, *d_b3;
    
    float* d_input, *d_hidden1, *d_hidden2, *d_output;
    float* d_dhidden1, *d_dhidden2, *d_doutput;

    float *d_dw1, *d_db1, *d_dw2, *d_db2, *d_dw3, *d_db3;

public:
    SimpleMNISTNet() {
        CUDA_CHECK(cudaMalloc(&d_w1, input_size * hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b1, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w2, hidden1_size * hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b2, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w3, hidden2_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b3, output_size * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden1, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden2, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_dhidden1, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dhidden2, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_doutput, output_size * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_dw1, input_size * hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_db1, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw2, hidden1_size * hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_db2, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dw3, hidden2_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_db3, output_size * sizeof(float)));

        initialize_weights();
        reset_gradients();
    }

    ~SimpleMNISTNet() {
        cudaFree(d_w1); cudaFree(d_b1);
        cudaFree(d_w2); cudaFree(d_b2);
        cudaFree(d_w3); cudaFree(d_b3);
        cudaFree(d_input); cudaFree(d_hidden1);
        cudaFree(d_hidden2); cudaFree(d_output);
        cudaFree(d_dhidden1); cudaFree(d_dhidden2); cudaFree(d_doutput);
        cudaFree(d_dw1); cudaFree(d_db1);
        cudaFree(d_dw2); cudaFree(d_db2);
        cudaFree(d_dw3); cudaFree(d_db3);
    }

private:
    void initialize_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        std::vector<float> h_w1(input_size * hidden1_size);
        std::vector<float> h_b1(hidden1_size);
        std::vector<float> h_w2(hidden1_size * hidden2_size);
        std::vector<float> h_b2(hidden2_size);
        std::vector<float> h_w3(hidden2_size * output_size);
        std::vector<float> h_b3(output_size);

        for (auto& val : h_w1) val = dist(gen);
        for (auto& val : h_b1) val = dist(gen);
        for (auto& val : h_w2) val = dist(gen);
        for (auto& val : h_b2) val = dist(gen);
        for (auto& val : h_w3) val = dist(gen);
        for (auto& val : h_b3) val = dist(gen);

        CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w3, h_w3.data(), h_w3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3, h_b3.data(), h_b3.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void reset_gradients() {
        CUDA_CHECK(cudaMemset(d_dw1, 0, input_size * hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db1, 0, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dw2, 0, hidden1_size * hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db2, 0, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dw3, 0, hidden2_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_db3, 0, output_size * sizeof(float)));
    }

public:
    float forward(const float* input, int label) {
        CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice));

        linear_forward_kernel<<<(hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_input, d_w1, d_b1, d_hidden1, input_size, hidden1_size);

        linear_forward_kernel<<<(hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_hidden1, d_w2, d_b2, d_hidden2, hidden1_size, hidden2_size);

        linear_forward_kernel<<<(output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_hidden2, d_w3, d_b3, d_output, hidden2_size, output_size);

        softmax_kernel<<<1, 1>>>(d_output, output_size);

        float loss;
        float* d_loss;
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        cross_entropy_loss_kernel<<<1, 1>>>(d_output, label, d_loss);
        CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_loss));

        CUDA_CHECK(cudaDeviceSynchronize());
        return loss;
    }

    void backward(int label, float learning_rate) {
        CUDA_CHECK(cudaMemset(d_dhidden1, 0, hidden1_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_dhidden2, 0, hidden2_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_doutput, 0, output_size * sizeof(float)));

        backward_output_kernel<<<(output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_output, label, d_doutput, output_size);

        linear_backward_kernel<<<(output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_doutput, d_hidden2, d_w3, d_dhidden2, d_dw3, d_db3, hidden2_size, output_size);

        linear_backward_kernel<<<(hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_dhidden2, d_hidden1, d_w2, d_dhidden1, d_dw2, d_db2, hidden1_size, hidden2_size);

        linear_backward_kernel<<<(hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_dhidden1, d_input, d_w1, d_dw1, d_dw1, d_db1, input_size, hidden1_size);

        update_weights_kernel<<<(input_size * hidden1_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_w1, d_dw1, input_size * hidden1_size, d_b1, d_db1, hidden1_size, learning_rate);
        
        update_weights_kernel<<<(hidden1_size * hidden2_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_w2, d_dw2, hidden1_size * hidden2_size, d_b2, d_db2, hidden2_size, learning_rate);
        
        update_weights_kernel<<<(hidden2_size * output_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_w3, d_dw3, hidden2_size * output_size, d_b3, d_db3, output_size, learning_rate);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int predict(const float* input) {
        forward(input, 0);
        
        std::vector<float> output(output_size);
        CUDA_CHECK(cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        int predicted = 0;
        for (int i = 1; i < output_size; i++) {
            if (output[i] > output[predicted]) {
                predicted = i;
            }
        }
        return predicted;
    }
};

// dum dum training loop
void train_mnist_simple() {
    auto train_data = load_mnist_dataset("../data/MNIST/raw/train-images-idx3-ubyte", "../data/MNIST/raw/train-labels-idx1-ubyte");
    
    SimpleMNISTNet model;
    DataLoader loader(train_data, 128, true);  // Batch size 128
    
    const int epochs = 5;
    const float learning_rate = 0.01f;
    
    std::cout << "Starting training..." << std::endl;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        int correct = 0;
        int total = 0;
        
        loader.reset();
        
        while (loader.has_next()) {
            auto batch = loader.next_batch();
            
            for (auto* sample : batch) {
                float loss = model.forward(sample->pixels.data(), sample->label);
                total_loss += loss;
                
                model.backward(sample->label, learning_rate);
                
                int predicted = model.predict(sample->pixels.data());
                if (predicted == sample->label) {
                    correct++;
                }
                total++;
            }
            
            if (total % 1000 == 0) {
                std::cout << "Processed " << total << " samples" << std::endl;
            }
        }
        
        float accuracy = static_cast<float>(correct) / total * 100.0f;
        float avg_loss = total_loss / total;
        
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << " - Loss: " << avg_loss 
                  << " - Accuracy: " << accuracy << "%" << std::endl;
    }
    
    std::cout << "Training completed!" << std::endl;
}

int main() {
    train_mnist_simple();
    return 0;
}