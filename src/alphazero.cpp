module;

#include <torch/torch.h>

export module damathzero;

import std;

export import :arena;
export import :config;
export import :game;
export import :memory;
export import :mcts;
export import :node;
export import :storage;

namespace DamathZero {

export template <Concepts::Game Game>
class AlphaZero {
  using Network = Game::Network;
  using State = Game::State;

 public:
  AlphaZero(Config config, std::shared_ptr<Network> model,
            std::shared_ptr<torch::optim::Optimizer> optimizer,
            std::random_device& rd)
      : config_(config), model_(model), optimizer_(optimizer), rd_(rd) {}

  auto learn() -> void {
    auto arena = Arena<Game>(config_);
    auto best_model_index = 0;

    for (auto i : std::views::iota(0, config_.num_iterations)) {
      std::println("Iteration: {}", i + 1);
      auto memory = Memory{config_, rd_};

      for (auto _ : std::views::iota(0, config_.num_self_play_iterations)) {
        generate_self_play_data(memory, model_);
      }

      for (auto _ : std::views::iota(0, config_.num_training_epochs)) {
        train(memory);
      }

      save_model(model_, i);

      auto best_model = read_model(best_model_index);

      auto results = arena.play(model_, best_model,
                                config_.num_model_evaluation_iterations,
                                /*num_simulations=*/1000);

      auto did_win =
          results.wins + results.draws >
          0.7 * static_cast<double>(config_.num_model_evaluation_iterations);

      std::println(
          "Trained model {} against the best model with {} wins, {} draws, "
          "and {} losses.",
          did_win ? "won" : "lost", results.wins, results.draws,
          results.losses);

      if (did_win) {
        best_model_index = i;
      }
    }
  }

 private:
  auto save_model(std::shared_ptr<Network> model, int checkpoint) const
      -> void {
    torch::serialize::OutputArchive output_model_archive;
    model->to(torch::kCPU);
    model->save(output_model_archive);
    output_model_archive.save_to(std::format("models/model_{}.pt", checkpoint));
  };

  auto read_model(int checkpoint) const -> std::shared_ptr<Network> {
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(std::format("models/model_{}.pt", checkpoint));
    auto model = std::make_shared<Network>();
    model->load(input_archive);
    model->to(config_.device);
    return model;
  }

  auto train(Memory& memory) {
    std::println("Database size {}", memory.size());
    namespace F = torch::nn::functional;
    model_->train();
    model_->to(config_.device);

    memory.shuffle();

    for (size_t i = 0; i < memory.size(); i += config_.batch_size) {
      auto [feature, target_value, target_policy] = memory.sample_batch(i);
      auto [out_value, out_policy] = model_->forward(feature);

      auto loss = F::mse_loss(out_value, target_value) +
                  F::cross_entropy(out_policy, target_policy);

      optimizer_->zero_grad();
      loss.backward();
      optimizer_->step();
    }
  }

  auto run_actor(Memory& memory, std::shared_ptr<Network> model) {
    auto mcts = MCTS<Game>{config_};

    auto statistics = std::vector<std::tuple<State, torch::Tensor>>();
    auto state = Game::initial_state();

    while (true) {
      auto action_probs = mcts.search(state, model);

      statistics.emplace_back(state, action_probs);

      auto action = torch::multinomial(action_probs, 1).template item<Action>();

      auto new_state = Game::apply_action(state, action);
      auto terminal_value = Game::terminal_value(new_state, action);

      if (terminal_value.has_value()) {
        auto value = *terminal_value;
        for (auto [hist_state, hist_probs] : statistics) {
          auto hist_value = hist_state.player == state.player ? value : -value;
          memory.append(Game::encode_state(hist_state),
                        torch::tensor({hist_value}, torch::kFloat32),
                        hist_probs);
        }
        break;
      }

      state = new_state;
    }
  }

  auto generate_self_play_data(Memory& memory, std::shared_ptr<Network> model)
      -> void {
    model->eval();
    auto threads = std::vector<std::thread>();

    for (auto _ : std::views::iota(0, config_.num_actors)) {
      threads.emplace_back(
          [this, &memory, model]() { run_actor(memory, model); });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

 private:
  Config config_;
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
};

}  // namespace DamathZero
