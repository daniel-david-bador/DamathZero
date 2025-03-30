module;

#include <torch/torch.h>

export module damathzero;

import std;

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
      : config_(config),
        model_(model),
        optimizer_(optimizer),
        rd_(rd),
        mcts_(config) {}

  auto learn() -> void {
    auto best_model_index = 0;

    for (auto i : std::views::iota(0, config_.num_iterations)) {
      std::println("Iteration: {}", i + 1);
      auto memory = Memory{config_, rd_};

      for (auto _ : std::views::iota(0, config_.num_self_play_iterations)) {
        memory.merge(generate_self_play_data());
      }

      for (auto _ : std::views::iota(0, config_.num_training_epochs))
        train(memory);

      save_model(model_, i);

      auto best_model = read_model(best_model_index);
      if (compete(best_model))
        best_model_index = i;
    }
  }

  auto search(Game::State state, int num_simulations) -> torch::Tensor {
    return mcts_.search(state, model_, num_simulations);
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

  auto generate_self_play_data() -> Memory {
    model_->eval();

    auto statistics = std::vector<std::tuple<State, torch::Tensor>>();
    auto state = Game::initial_state();

    while (true) {
      auto action_probs = mcts_.search(state, model_);

      statistics.emplace_back(state, action_probs);

      auto action = torch::multinomial(action_probs, 1).template item<Action>();

      auto new_state = Game::apply_action(state, action);
      auto terminal_value = Game::terminal_value(new_state, action);

      if (terminal_value.has_value()) {
        auto memory = Memory{config_, rd_};
        auto value = *terminal_value;
        for (auto [hist_state, hist_probs] : statistics) {
          auto hist_value = hist_state.player == state.player ? value : -value;
          memory.append(Game::encode_state(hist_state),
                        torch::tensor({hist_value}, torch::kFloat32),
                        hist_probs);
        }
        return memory;
      }

      state = new_state;
    }
  }

  auto compete(std::shared_ptr<Network> best_model) -> bool {
    model_->eval();
    best_model->eval();

    auto trained_model_player = Player::First;

    auto wins = 0.0;
    auto draws = 0.0;
    auto loss = 0.0;

    for (auto _ :
         std::views::iota(0, config_.num_model_evaluation_iterations)) {
      auto state = Game::initial_state();

      auto value = 0.0;

      while (true) {
        torch::Tensor action_probs;
        if (state.player == trained_model_player)
          action_probs = mcts_.search(state, model_);
        else
          action_probs = mcts_.search(state, best_model);

        auto action = torch::argmax(action_probs).template item<Action>();

        auto new_state = Game::apply_action(state, action);
        auto terminal_value = Game::terminal_value(new_state, action);

        if (terminal_value.has_value()) {
          if (value == 0) {
            draws += 1;
          } else {
            if (state.player == trained_model_player)
              wins += 1;
            else
              loss += 1;
          }
          break;
        }

        state = new_state;
      }
    }

    auto did_win =
        wins + draws >
        0.7 * static_cast<double>(config_.num_model_evaluation_iterations);
    if (did_win) {
      std::println(
          "Trained model won against the best model with {} wins, {} draws, "
          "and {} losses.",
          wins, draws, loss);
    } else {
      std::println(
          "Trained model did not win against the best model with {} wins, {} "
          "draws, and {} losses.",
          wins, draws, loss);
    }
    return did_win;
  }

 private:
  Config config_;
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
  MCTS<Game> mcts_;
};

}  // namespace DamathZero
