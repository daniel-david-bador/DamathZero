# DamatZero

DamatZero is an AlphaZero implementation in C++ with the PyTorch frontend applied to the game of Damath (a mathematical variant of checkers). It is designed to be a fast and efficient engine for playing Damath, leveraging the principles of deep reinforcement learning and Monte Carlo Tree Search (MCTS).

## Requirements

- C++23 compatible compiler
- PyTorch C++ frontend
- CMake 3.11 or higher
- CUDA (optional, for GPU support)

## Building the Project

1. Download the source code.
2. Ensure you have the required dependencies installed.
3. Create a build directory:
   ```bash
   mkdir build
   ```
4. Run CMake to configure the project:
   ```bash
   cmake -S . -B build
   ```
5. Build the project:
   ```bash
    cmake --build build
   ```

## Running the Engine

To run the DamatZero engine, execute the following command in the terminal:

```bash
./build/DamathZeroApp <path_to_model.pt>
```

## Running the Trainer

If you do not have a CUDA-enabled GPU, you can run the trainer in CPU mode. Change the `device` in the `src/train.cpp` file to `torch::kCPU`. To train the model, use the following command:

```bash
./build/DamathZeroTrainer
```