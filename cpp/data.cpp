// | File                    | Description            |
// | ----------------------- | ---------------------- |
// | train-images-idx3-ubyte | 60,000 training images |
// | train-labels-idx1-ubyte | 60,000 training labels |
// | t10k-images-idx3-ubyte  | 10,000 test images     |
// | t10k-labels-idx1-ubyte  | 10,000 test labels     |

#include "data.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

uint32_t read_uint32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<MNISTSample> load_mnist_dataset(
    const std::string& image_path,
    const std::string& label_path
) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    
    if (!image_file.is_open()) throw std::runtime_error("Cannot open image file");
    if (!label_file.is_open()) throw std::runtime_error("Cannot open label file");

    uint32_t magic_images = read_uint32(image_file);
    uint32_t num_images = read_uint32(image_file);
    uint32_t rows = read_uint32(image_file);
    uint32_t cols = read_uint32(image_file);

    uint32_t magic_labels = read_uint32(label_file);
    uint32_t num_labels = read_uint32(label_file);

    if (magic_images != 0x00000803 || magic_labels != 0x00000801)
        throw std::runtime_error("Invalid MNIST magic number");

    if (num_images != num_labels)
        throw std::runtime_error("Image and label count mismatch");

    std::vector<MNISTSample> dataset(num_images);
    std::vector<unsigned char> buffer(IMAGE_SIZE);
    
    for (uint32_t i = 0; i < num_images; ++i) {
        image_file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
        
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);

        MNISTSample sample;
        sample.label = label;
        for (int j = 0; j < IMAGE_SIZE; ++j)
            sample.pixels[j] = buffer[j] / 255.0f;

        dataset[i] = std::move(sample);
    }

    return dataset;
}

DataLoader::DataLoader(std::vector<MNISTSample>& data, size_t batch_sz, bool shuf)
    : dataset(data), batch_size(batch_sz), shuffle(shuf), current_idx(0), rng(std::random_device{}()) {
    indices.resize(dataset.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
}

std::vector<MNISTSample*> DataLoader::next_batch() {
    if (current_idx >= dataset.size()) return {};
    
    std::vector<MNISTSample*> batch;
    size_t end = std::min(current_idx + batch_size, dataset.size());
    
    for (size_t i = current_idx; i < end; ++i)
        batch.push_back(&dataset[indices[i]]);
    
    current_idx = end;
    return batch;
}

void DataLoader::reset() {
    current_idx = 0;
    if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
}

bool DataLoader::has_next() const {
    return current_idx < dataset.size();
}

size_t DataLoader::num_batches() const {
    return (dataset.size() + batch_size - 1) / batch_size;
}



//g++ -std=c++20 data.cpp -o mnist