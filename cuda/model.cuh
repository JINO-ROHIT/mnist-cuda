#ifndef MODEL_HEADERS
#define MODEL_HEADERS

#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "data.hpp"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0) 


class CUDALinear{
    private:
            int in_features, out_features;

            float *d_weight;
            float *d_bias;
            float *d_weight_grad;
            float *d_bias_grad;
            float *d_input_cache;

            size_t max_batch_size;
    
    public:
            CUDALinear(int in_features, int out_features, int max_batch_size); //constructor
            ~CUDALinear(); //destructor;
            void forward(const float* x);
            void backward(const float* grad_output);
            void zero_grad();
            void update(float* learning_rate);
}

class CUDANet{
    private:
            CUDALinear* l1;
            CUDALinear* l2;
            CUDALinear* l3;
            CUDALinear* l4;

            size_t hidden_size;
            size_t max_batch_size;

            // buffers for intermediate activations
            float* d_out1_pre_relu;
            float* d_out1;
            float* d_out2_pre_relu;
            float* d_out2;
            float* d_out3_pre_relu;
            float* d_out3;
            float* d_logits;
            
            // buffers for gradients
            float* d_grad_logits;
            float* d_grad_out3;
            float* d_grad_out2;
            float* d_grad_out1;
            
            // buffer for accuracy computation
            float* h_logits;
    public:
            CUDANet(size_t hidden_size);
            ~CUDANet();

            void forward(const float* d_input);
            void backward(const int* d_labels);
            void zero_grad();
            void update(float learning_rate);
            
            float compute_accuracy(const std::vector<MNISTSample*>& batch);
            float train_batch(const std::vector<MNISTSample*>& batch, float learning_rate);
}

#endif