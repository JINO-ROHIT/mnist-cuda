#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "model.cuh"

//kernels

__global__ void matrix_mul(float* A, 
                    float* B, 
                    float* C,
                    int M,
                    int N,
                    int K){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;

        for(int i = 0; i < K, i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_backward_kernel(
    float* grad_output, const float* pre_relu_data, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        if (pre_relu_data[idx] <= 0.0f) {
            grad_output[idx] = 0.0f;
        }
    }
}

CUDALinear::CUDALinear(int input, int output, int batch_size) : in_features(input), out_features(output), max_batch_size(batch_size){
    CHECK_CUDA(cudaMalloc(&d_weight, sizeof(float) * in_features * out_features));
    CHECK_CUDA(cudaMalloc(&d_bias, sizeof(float) * out_features));

    //for gradients
    CHECK_CUDA(cudaMalloc(&d_weight_grad, sizeof(float) * in_features * out_features));
    CHECK_CUDA(cudaMalloc(&d_bias_grad, sizeof(float) * out_features));

    //for input cache bc we have [batch_size,in_features]
    CHECK_CUDA(cudaMalloc(&d_input_cache, sizeof(float) * max_batch_size * in_features);

    //initialize weights
    std::vector<float> h_weight(in_features * out_features);
    float limit = std::sqrt(6.0f / (in_features + out_features)); // xavier init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-limit, limit);
    for(auto& w: h_weight) w = dist(gen);

    //copy these values to gpu
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), sizeof(float) * in_features * out_features), cudaMemcpyHostToDevice); //(device ptr, host ptr, size)

    // initialize bias to zero
    cudaMemset(d_bias, 0, out_features * sizeof(float));
    cudaMemset(d_weight_grad, 0, in_features * out_features * sizeof(float));
    cudaMemset(d_bias_grad, 0, out_features * sizeof(float));
}

CUDALinear::~CUDALinear(){
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_weight_grad);
    cudaFree(d_bias_grad);
    cudaFree(d_input_cache);
}