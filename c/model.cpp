// nn.Linear(784, hidden_size),
// nn.ReLU(),
// nn.Linear(hidden_size, hidden_size),
// nn.ReLU(),
// nn.Linear(hidden_size, hidden_size),
// nn.ReLU(),
// nn.Linear(hidden_size, 10),

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

inline float relu(float x){
    return x > 0 ? x : 0;
}

//check correctness
inline float log_softmax(const std::vector<float>& logits, int idx) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (float l : logits) sum_exp += std::exp(l - max_logit);
    return logits[idx] - max_logit - std::log(sum_exp);
}