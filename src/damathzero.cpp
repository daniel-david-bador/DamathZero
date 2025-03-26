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

export template <typename Network>
class DamathZero {
 public:
  DamathZero(Config config, std::shared_ptr<Network> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer,
             std::random_device& rd)
      : config_(config),
        model_(model),
        optimizer_(optimizer),
        rd_(rd),
        mcts_{config} {}

  auto learn() -> void {
    for (auto _ : std::views::iota(0, config_.NumIterations)) {
      auto memory = Memory{config_, rd_};

      model_->eval();
      for (auto _ : std::views::iota(0, config_.NumSelfPlayIterations))
        memory.merge(generate_self_play_data());

      model_->train();
      for (auto _ : std::views::iota(0, config_.NumTrainingEpochs))
        train(memory);

      // torch::save(model_, std::format("models/model_{}.pt", i));
      // torch::save(optimizer_, std::format("optimizer_{}.pt", i));
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
    auto statistics = std::vector<std::tuple<Board, torch::Tensor, Player>>();
    auto player = Player{1};
    auto board = Game::initial_board();

    while (true) {
      auto neutral_state = Game::change_perspective(board, player);
      auto action_probs = mcts_.search(neutral_state, model_);

      statistics.emplace_back(neutral_state, action_probs, player);

      auto action = torch::multinomial(action_probs, 1).template item<Action>();

      auto [new_board, new_player] = Game::apply_action(board, action, player);
      auto [value, is_terminal] =
          Game::get_value_and_terminated(new_board, action);

      if (is_terminal) {
        auto memory = Memory{config_, rd_};
        for (auto [hist_board, hist_probs, hist_player] : statistics) {
          auto hist_value = hist_player == player ? value : -value;
          memory.append(torch::tensor(hist_board, torch::kFloat32),
                        torch::tensor({hist_value}, torch::kFloat32),
                        hist_probs);
        }
        return memory;
      }

      board = new_board;
      player = new_player;
    }
  }

 private:
  Config config_;
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
  MCTS mcts_;
};

}  // namespace DamathZero
