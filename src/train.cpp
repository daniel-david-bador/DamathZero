import dz;
import std;

auto main(int argc, char** argv) -> int {
  auto damathzero = dz::DamathZero{{
      .batch_size = 64,
      .num_training_epochs = 4,
      .num_training_iterations = 10,
      .num_self_play_actors = 6,
      .num_self_play_iterations = 100,
      .num_self_play_simulations = 60,
      .num_evaluation_actors = 5,
      .num_evaluation_iterations = 10,
      .num_evaluation_simulations = 1000,
      .device = dz::DeviceType::CPU,
  }};

  auto model_config = dz::Model::Config{
      .action_size = dz::Game::ActionSize,
      .num_blocks = 10,
      .num_attention_head = 4,
      .embedding_dim = 64,
      .mlp_hidden_size = 128,
      .mlp_dropout_prob = 0.1,
  };

  std::optional<std::shared_ptr<dz::Model>> previous_model = std::nullopt;
  if (argc > 1) {
    previous_model = dz::load_model(argv[1], model_config);
  }

  auto model = damathzero.learn(model_config, previous_model);
  dz::save_model(model, "models/best_model.pt");
}
