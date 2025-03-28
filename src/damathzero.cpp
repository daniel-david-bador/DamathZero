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
    auto best_model_index = 0;
    for (auto i : std::views::iota(0, config_.NumIterations)) {
      std::println("Iteration: {}", i);
      auto memory = Memory{config_, rd_};

      model_->eval();
      for (auto _ : std::views::iota(0, config_.NumSelfPlayIterations))
        memory.merge(generate_self_play_data());

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

  auto compete(std::shared_ptr<Network> best_model) -> bool {
    double wins = 0.0;
    double draws = 0.0;
    double loss = 0.0;

    for (auto _ : std::views::iota(0, config_.NumModelEvaluationIterations)) {
      auto player = Player{1};
      auto board = Game::initial_board();

      double value = 0.0;
      bool is_terminal = false;

      while (not is_terminal) {
        auto neutral_state = Game::change_perspective(board, player);
        torch::Tensor action_probs;
        if (player == 1)
          action_probs = mcts_.search(neutral_state, model_);
        else
          action_probs = mcts_.search(neutral_state, best_model);

        auto action = torch::argmax(action_probs).template item<Action>();

        auto [new_board, new_player] =
            Game::apply_action(board, action, player);
        std::tie(value, is_terminal) =
            Game::get_value_and_terminated(new_board, action);

        if (is_terminal) {
          if (player == 1) {
            if (value == 0)
              draws += 1;
            else
              wins += 1;
          } else {
            loss += 1;
          }
        }

        board = new_board;
        player = new_player;
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
  MCTS mcts_;
};

}  // namespace DamathZero
