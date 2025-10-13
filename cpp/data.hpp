#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <random>

constexpr int IMAGE_ROWS = 28;
constexpr int IMAGE_COLS = 28;
constexpr int IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS;

struct MNISTSample {
    std::array<float, IMAGE_SIZE> pixels;
    uint8_t label;
};

uint32_t read_uint32(std::ifstream& file);

std::vector<MNISTSample> load_mnist_dataset(
    const std::string& image_path,
    const std::string& label_path
);

class DataLoader {
private:
    std::vector<MNISTSample>& dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    size_t current_idx;
    std::mt19937 rng;

public:
    DataLoader(std::vector<MNISTSample>& data, size_t batch_sz, bool shuf = true);
    std::vector<MNISTSample*> next_batch();
    void reset();
    bool has_next() const;
    size_t num_batches() const;
};

#endif