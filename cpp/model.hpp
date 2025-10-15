#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include "data.hpp"

inline float relu(float x);
inline float relu_derivative(float x);

struct Linear {
    int in_features, out_features;
    std::vector<float> weight;
    std::vector<float> bias;
    
    std::vector<float> weight_grad;
    std::vector<float> bias_grad;
    
    // cache for backward pass
    std::vector<float> input_cache;

    Linear(int inp, int out);
    std::vector<float> forward(const std::vector<float>& x);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void zero_grad();
    void update(float learning_rate);
};

struct MLPNet {
    Linear l1, l2, l3, l4;
    
    // cache for backward pass
    std::vector<float> out1, out2, out3, logits;
    std::vector<float> out1_pre_relu, out2_pre_relu, out3_pre_relu;
    
    MLPNet(int hidden_size = 512);
    
    std::vector<float> forward(const std::vector<float>& x);
    void backward(const std::vector<float>& x, int target_label);
    void zero_grad();
    void update(float learning_rate);
    
    int predict(const std::vector<float>& log_probs) const;
    float compute_accuracy(const std::vector<MNISTSample*>& batch);
    float train_batch(const std::vector<MNISTSample*>& batch, float learning_rate);
};

#endif