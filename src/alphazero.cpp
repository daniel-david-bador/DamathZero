module;

#include <torch/torch.h>

export module damathzero;

import std;

export import :config;
export import :game;
export import :memory;
export import :mcts;
export import :network;
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
        mcts_{config} {}

  auto learn() -> void {
    auto best_model_index = 0;
    for (auto i : std::views::iota(0, config_.NumIterations)) {
      std::println("Iteration: {}", i);
      auto memory = Memory{config_, rd_};

      model_->eval();
      for (auto _ : std::views::iota(0, config_.NumSelfPlayIterations)) {
        memory.merge(generate_self_play_data());
      }

      model_->train();
      for (auto _ : std::views::iota(0, config_.NumTrainingEpochs))
        train(memory);

      torch::serialize::OutputArchive output_model_archive;
      model_->to(torch::kCPU);
      model_->save(output_model_archive);
      output_model_archive.save_to(std::format("models/model_{}.pt", i));

      torch::serialize::InputArchive input_archive;
      input_archive.load_from(
          std::format("models/model_{}.pt", best_model_index));
      auto best_model = std::make_shared<Network>();
      best_model->load(input_archive);
      best_model->to(torch::kCPU);

      model_->eval();
      best_model->eval();
      if (compete(best_model))
        best_model_index = i;
    }
  }

  auto train(Memory& memory) {
    namespace F = torch::nn::functional;
    memory.shuffle();
    for (size_t i = 0; i < memory.size(); i += config_.BatchSize) {
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
          auto hist_value =
              hist_state.player == new_state.player ? value : -value;
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
    // Player 1 plays using the trained model
    // Player 2 plays using the best model so far

    double wins = 0.0;
    double draws = 0.0;
    double loss = 0.0;

    for (auto _ : std::views::iota(0, config_.NumModelEvaluationIterations)) {
      auto state = Game::initial_state();

      double value = 0.0;
      bool is_terminal = false;

      while (not is_terminal) {
        torch::Tensor action_probs;
        if (state.player == 1)
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
            if (state.player == 1)
              wins += 1;
            else
              loss += 1;
          }
          is_terminal = true;
        }

        state = new_state;
      }
    }

    auto win = wins + draws >
               0.7 * static_cast<double>(config_.NumModelEvaluationIterations);
    if (win) {
      // model_.swap(pretrained_model);
      std::println("Trained model won against the best model {}:{}:{}!", wins,
                   draws, loss);
    } else {
      std::println(
          "Trained model did not win against the best model {}:{}:{}...", wins,
          draws, loss);
    }
    return win;
  }

 private:
  Config config_;
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
  MCTS<Game> mcts_;
};

}  // namespace DamathZero
