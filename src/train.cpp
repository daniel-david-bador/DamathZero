#include <memory>
#include <optional>

#include "damathzero/dz.hpp"

auto main(int argc, char** argv) -> int {
  auto damathzero = dz::DamathZero{{
      .temperature = 1.25,
      .batch_size = 1024,
      .num_iterations = 20,
      .num_training_epochs = 10,
      .num_self_play_actors = 50,
      .num_self_play_games = 100,
      .num_self_play_simulations = 100,
      .num_evaluation_games = 100,
      .num_evaluation_simulations = 100,
      .device = dz::DeviceType::CUDA,
  }};

  auto model_config = dz::Model::Config{
      .action_size = dz::Game::ActionSize,
      .num_blocks = 16,
      .num_attention_head = 16,
      .embedding_dim = 256,
      .mlp_hidden_size = 512,
      .mlp_dropout_prob = 0.1,
  };

  std::optional<std::shared_ptr<dz::Model>> previous_model = std::nullopt;
  if (argc > 1) {
    previous_model = dz::load_model(argv[1], model_config);
  }

  auto model = damathzero.learn(model_config, previous_model);
  dz::save_model(model, "models/best_model.pt");
}
