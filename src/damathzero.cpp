module;

#include <torch/torch.h>

export module damathzero;

import std;

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
  DamathZero(std::shared_ptr<Network> model,
             std::shared_ptr<torch::optim::Optimizer> optimizer,
             std::random_device& rd)
      : model_(model), optimizer_(optimizer), rd_(rd) {}

  auto learn() -> void {
    auto memory = Memory{rd_};

    for (auto _ : std::views::iota(0, Config::NumIterations)) {
      model_->eval();
      for (auto _ : std::views::iota(0, Config::NumSelfPlayIterations))
        memory.merge(generate_self_play_data());

      model_->train();
      for (auto _ : std::views::iota(0, Config::NumTrainingEpochs))
        train(memory);

      // torch::save(model_, std::format("models/model_{}.pt", i));
      // torch::save(optimizer_, std::format("optimizer_{}.pt", i));
    }
  }

  auto train(Memory& memory) {
    namespace F = torch::nn::functional;
    for (size_t i = 0; i < memory.size(); i += Config::BatchSize) {
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
      auto network = std::make_shared<Network>();
      auto mcts = MCTS{};

      while (true) {
          auto neutral_state = Game::change_perspective(board, player);
          auto action_probs = mcts.search(neutral_state, network);

          statistics.emplace_back(neutral_state, action_probs, player);

          auto action = torch::multinomial(action_probs, 1).template item<Action>();

          auto [new_board, new_player] = Game::apply_action(board, action, player);
          auto [value, is_terminal] = Game::get_value_and_terminated(new_board, action);

          if (is_terminal) {
              auto memory = Memory{rd_};
              for (auto [hist_board, hist_probs, hist_player] : statistics) {
                  auto terminal_value = hist_player == player ? value : -value;
                  memory.append(torch::tensor(hist_board, torch::kFloat32),  torch::tensor({terminal_value}, torch::kFloat32), hist_probs);
              }
              return memory;
          }

          board = new_board;
          player = new_player;
      }
  }

 private:
  std::shared_ptr<Network> model_;
  std::shared_ptr<torch::optim::Optimizer> optimizer_;
  std::random_device& rd_;
};

}  // namespace DamathZero
