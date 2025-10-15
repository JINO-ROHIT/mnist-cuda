#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

inline float relu(float x) {
    return x > 0 ? x : 0;
}

inline float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

Linear::Linear(int inp, int out) : in_features(inp), out_features(out) {
    std::mt19937 gen(42);
    float limit = std::sqrt(6.0f / (in_features + out_features)); // xavier initialization
    std::uniform_real_distribution<float> dist(-limit, limit);

    weight.resize(out_features * in_features);
    bias.resize(out_features, 0.0f);
    weight_grad.resize(out_features * in_features, 0.0f);
    bias_grad.resize(out_features, 0.0f);

    for (auto& w : weight) w = dist(gen);
}

std::vector<float> Linear::forward(const std::vector<float>& x) {
    input_cache = x; // cache input for backward pass
    
    std::vector<float> out(out_features, 0.0f);
    for (int i = 0; i < out_features; ++i) {
        float sum = bias[i];
        for (int j = 0; j < in_features; ++j) {
            sum += weight[i * in_features + j] * x[j];
        }
        out[i] = sum;
    }
    return out;
}

std::vector<float> Linear::backward(const std::vector<float>& grad_output) {
    std::vector<float> grad_input(in_features, 0.0f);
    
    for (int i = 0; i < out_features; ++i) {
        bias_grad[i] += grad_output[i];
        for (int j = 0; j < in_features; ++j) {
            weight_grad[i * in_features + j] += grad_output[i] * input_cache[j];
            grad_input[j] += grad_output[i] * weight[i * in_features + j];
        }
    }
    
    return grad_input;
}

void Linear::zero_grad() {
    std::fill(weight_grad.begin(), weight_grad.end(), 0.0f);
    std::fill(bias_grad.begin(), bias_grad.end(), 0.0f);
}

void Linear::update(float learning_rate) {
    for (size_t i = 0; i < weight.size(); ++i) {
        weight[i] -= learning_rate * weight_grad[i];
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learning_rate * bias_grad[i];
    }
}

MLPNet::MLPNet(int hidden_size)
    : l1(784, hidden_size),
      l2(hidden_size, hidden_size),
      l3(hidden_size, hidden_size),
      l4(hidden_size, 10) {}

std::vector<float> MLPNet::forward(const std::vector<float>& x) {
    out1_pre_relu = l1.forward(x);
    out1 = out1_pre_relu;
    std::transform(out1.begin(), out1.end(), out1.begin(), relu);

    out2_pre_relu = l2.forward(out1);
    out2 = out2_pre_relu;
    std::transform(out2.begin(), out2.end(), out2.begin(), relu);

    out3_pre_relu = l3.forward(out2);
    out3 = out3_pre_relu;
    std::transform(out3.begin(), out3.end(), out3.begin(), relu);

    logits = l4.forward(out3);

    return logits;
}