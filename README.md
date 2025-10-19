## MNIST-CUDA

This project is an attempt to write the mnist training loop in pure CUDA C++ and check for performance gains over ML libraries like pytorch.

### Get started

1. Install the libraries

```bash
uv sync
```

2. Run torch baseline on cpu

```bash
python3 mnist_benchmark.py --epochs 5 --device cpu
```

3. Run torch baseline on mps

```bash
python3 mnist_benchmark.py --epochs 5 --device mps
```

4. Run Native C++ baseline.

```cpp
g++ -std=c++20 data.cpp model.cpp main.cpp -o mnist_cpu
./mnist_cpu
```

5. Run CUDA C++ baseline.

```cpp
nvcc -std=c++20 -O3 model.cu data.cpp -o mnist_cuda
./mnist_cuda
```

### MNIST Training Benchmark (MLP, 5 Epochs)

| **Device**        | **Total Time (s)** | **Avg Time / Epoch (s)** | **Test Accuracy (%)** |
|--------------------|--------------------|---------------------------|-----------------------|
| PyTorch (CPU)      | 13.35              | 2.67                      | 94.57                 |
| PyTorch (MPS)      | 8.61               | 1.72                      | 94.96                 |
| C++ (Native)       | 303.30             | 63.42                     | 94.26                 |
| C++ (CUDA)         | 26                 | 5.12                      | 99.95                 |
