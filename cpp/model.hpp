#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include "data.hpp"

inline float relu(float x);
inline float log_softmax(const std::vector<float>& logits, int idx);

struct Linear {
    int in_features, out_features;
    std::vector<float> weight;
    std::vector<float> bias;

    Linear(int inp, int out);
    std::vector<float> forward(const std::vector<float>& x) const;
};

struct MLPNet {
    Linear l1, l2, l3, l4;
    
    MLPNet(int hidden_size = 512);
    
    std::vector<float> forward(const std::vector<float>& x) const;
    std::vector<std::vector<float>> forward_batch(const std::vector<MNISTSample*>& batch) const;
    int predict(const std::vector<float>& log_probs) const;
    float compute_accuracy(const std::vector<MNISTSample*>& batch) const;
};

#endif