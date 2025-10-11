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

### MNIST Training Benchmark (MLP, 5 Epochs)

| Device | Total Time (s) | Avg Time/Epoch (s) | Test Accuracy (%) |
|--------|----------------|------------------|-----------------|
| CPU    | 13.35          | 2.67             | 94.57           |
| MPS    | 8.61           | 1.72             | 94.96           |
| CUDA   |                |                  |                 |
