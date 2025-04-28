import dz;

auto main() -> int {
  auto damathzero = dz::DamathZero{dz::Config{
      .num_iterations = 2,
      .num_simulations = 10,
      .num_self_play_iterations_per_actor = 100,
      .num_actors = 5,
      .num_model_evaluation_iterations = 5,
      .num_model_evaluation_simulations = 100,
      .device = dz::DeviceType::CPU,
  }};

  auto model_config = dz::Model::Config{
      .action_size = dz::Game::ActionSize,
      .num_blocks = 2,
      .num_attention_head = 4,
      .embedding_dim = 64,
      .mlp_hidden_size = 128,
      .mlp_dropout_prob = 0.1,
  };

  auto model = damathzero.learn(model_config);

  dz::save_model(model, "models/best_model.pt");
}
