#include "model.hpp"
#include <cmath>
#include <algorithm>
#include <random>

inline float relu(float x) {
    return x > 0 ? x : 0;
}

inline float log_softmax(const std::vector<float>& logits, int idx) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float l : logits) sum_exp += std::exp(l - max_logit);
    return logits[idx] - max_logit - std::log(sum_exp);
}

Linear::Linear(int inp, int out) : in_features(inp), out_features(out) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    weight.resize(out_features * in_features);
    bias.resize(out_features);

    for (auto& w : weight) w = dist(gen);
    for (auto& b : bias) b = dist(gen);
}

std::vector<float> Linear::forward(const std::vector<float>& x) const {
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

MLPNet::MLPNet(int hidden_size)
    : l1(784, hidden_size),
      l2(hidden_size, hidden_size),
      l3(hidden_size, hidden_size),
      l4(hidden_size, 10) {}

std::vector<float> MLPNet::forward(const std::vector<float>& x) const {
    auto out1 = l1.forward(x);
    std::transform(out1.begin(), out1.end(), out1.begin(), relu);

    auto out2 = l2.forward(out1);
    std::transform(out2.begin(), out2.end(), out2.begin(), relu);

    auto out3 = l3.forward(out2);
    std::transform(out3.begin(), out3.end(), out3.begin(), relu);

    auto logits = l4.forward(out3);

    std::vector<float> log_probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        log_probs[i] = log_softmax(logits, static_cast<int>(i));

    return log_probs;
}

std::vector<std::vector<float>> MLPNet::forward_batch(const std::vector<MNISTSample*>& batch) const {
    std::vector<std::vector<float>> outputs;
    outputs.reserve(batch.size());
    
    for (const auto* sample : batch) {
        std::vector<float> input(sample->pixels.begin(), sample->pixels.end());
        outputs.push_back(forward(input));
    }
    
    return outputs;
}

int MLPNet::predict(const std::vector<float>& log_probs) const {
    return std::distance(log_probs.begin(), 
                       std::max_element(log_probs.begin(), log_probs.end()));
}

float MLPNet::compute_accuracy(const std::vector<MNISTSample*>& batch) const {
    int correct = 0;
    
    for (const auto* sample : batch) {
        std::vector<float> input(sample->pixels.begin(), sample->pixels.end());
        auto log_probs = forward(input);
        int predicted = predict(log_probs);
        
        if (predicted == sample->label) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / batch.size();
}