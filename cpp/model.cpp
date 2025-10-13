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

struct Linear {
    int in_features, out_features;
    std::vector<float> weight;
    std::vector<float> bias;

    Linear(int inp, int out) : in_features(inp), out_features(out) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

        weight.resize(out_features * in_features);
        bias.resize(out_features);

        for (auto& w: weight) w = dist(gen);
        for(auto& b: bias) b = dist(gen);
    }

    std::vector<float> forward(const std::vector<float>& x) const { 
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
    };

struct MLPNet {
    Linear l1, l2, l3, l4;
    MLPNet(int hidden_size = 512)
        : l1(784, hidden_size),
          l2(hidden_size, hidden_size),
          l3(hidden_size, hidden_size),
          l4(hidden_size, 10) {}

    std::vector<float> forward(const std::vector<float>& x) const {
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
};


int main() {
    MLPNet model(128);

    std::vector<float> input(784, 0.5f);

    auto output = model.forward(input);

    std::cout << "Output log-probabilities:\n";
    for (float o : output) std::cout << o << " ";
    std::cout << "\nPredicted class: "
              << std::distance(output.begin(), std::max_element(output.begin(), output.end()))
              << "\n";
}

// ensure you get log(10) ~ 2.xx, so everything is equally likely