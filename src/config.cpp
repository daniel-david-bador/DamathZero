module;

#include <torch/torch.h>

export module alphazero:config;

import std;

namespace AZ {

export struct Config {
  int32_t num_iterations = 10;
  int32_t num_simulations = 100;

  int32_t num_self_play_iterations_per_actor = 1000;
  int32_t num_actors = 6;

  int32_t num_training_epochs = 4;
  int32_t num_model_evaluation_iterations = 50;

  size_t batch_size = 64;

  float32_t C = 2.0;

  float32_t dirichlet_alpha = 0.3;
  float32_t dirichlet_epsilon = 0.25;

  float32_t temperature = 1.25;

  float32_t random_playout_percentage = 0.2;

  torch::DeviceType device;
};

}  // namespace AZ
