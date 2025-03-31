module;

#include <torch/torch.h>

export module damathzero:config;

import std;

namespace DamathZero {

export struct Config {
  int num_iterations = 10;
  int num_simulations = 100;

  int num_self_play_iterations = 1000;
  int num_training_epochs = 4;
  int num_model_evaluation_iterations = 50;

  int num_actors = 6;

  std::size_t batch_size = 64;

  double C = 2.0;

  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;

  torch::DeviceType device;
};

}  // namespace DamathZero
