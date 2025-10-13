// | File                    | Description            |
// | ----------------------- | ---------------------- |
// | train-images-idx3-ubyte | 60,000 training images |
// | train-labels-idx1-ubyte | 60,000 training labels |
// | t10k-images-idx3-ubyte  | 10,000 test images     |
// | t10k-labels-idx1-ubyte  | 10,000 test labels     |

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <stdint.h>
#include <string>


constexpr int IMAGE_ROWS = 28; //make it known at compile time
constexpr int IMAGE_COLS = 28;
constexpr int IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS;

struct MNISTSample {
    std::array<float, IMAGE_SIZE> pixels;
    uint8_t label; //unsigned 8 bit integer
};

uint32_t read_uint32(std::ifstream& file){
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

    // Read headers
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
        // Read pixels
        image_file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
        // Read label
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);

        MNISTSample sample;
        sample.label = label;
        for (int j = 0; j < IMAGE_SIZE; ++j)
            sample.pixels[j] = buffer[j] / 255.0f; // normalize

        dataset[i] = std::move(sample);
    }

    return dataset;
}


int main() {
    try {
        auto train_data = load_mnist_dataset(
            "../data/MNIST/raw/train-images-idx3-ubyte",
            "../data/MNIST/raw/train-labels-idx1-ubyte"
        );

        std::cout << "Loaded " << train_data.size() << " samples.\n";
        std::cout << "First label: " << static_cast<int>(train_data[0].label) << "\n";

        // first image
        const auto& sample = train_data[0];
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            if (i % IMAGE_COLS == 0) std::cout << "\n";
            std::cout << (sample.pixels[i] > 0.5f ? '#' : '.');
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}


//g++ -std=c++20 data.cpp -o mnist