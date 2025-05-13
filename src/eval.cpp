#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <indicators/dynamic_progress.hpp>
#include <indicators/progress_bar.hpp>
#include <print>
#include <random>
#include <ranges>

#include "damathzero/dz.hpp"

namespace opt = indicators::option;

constexpr auto path = "models_archive/best_models/model_{}.pt";

auto main(int, char**) -> int {
  std::vector<int> indices{0, 2, 3, 6, 11, 18};

  int32_t num_evaluation_games = 64;
  int32_t num_evaluation_simulations = 1000;

  dz::DeviceType device = dz::DeviceType::CUDA;

  indicators::DynamicProgress<indicators::ProgressBar> bars;

  auto config = dz::Model::Config{
      .action_size = dz::Game::ActionSize,
      .num_blocks = 16,
      .num_attention_head = 16,
      .embedding_dim = 256,
      .mlp_hidden_size = 512,
      .mlp_dropout_prob = 0.1,
  };

  auto best_model_index = indices.back();

  auto best_model = load_model(std::format(path, best_model_index), config);
  
  best_model->eval();
  best_model->to(device);

  for (const auto current_model_index : indices) {
    auto bar = std::make_unique<indicators::ProgressBar>(
        opt::BarWidth{50}, opt::ForegroundColor{colors[current_model_index % 6]},
        opt::ShowElapsedTime{true}, opt::ShowRemainingTime{true},
        opt::ShowPercentage{true},
        opt::MaxProgress{num_evaluation_games + 1},
        opt::PrefixText{
            std::format("v{} vs v{} ", current_model_index, best_model_index)},
        opt::PostfixText{"Evaluating Model | Wins: 0 - Draws: 0 - Losses: 0"},
        opt::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});
    auto bar_id = bars.push_back(std::move(bar));
    bars[bar_id].tick();

    auto current_model = load_model(std::format(path, current_model_index), config);

    current_model->eval();
    current_model->to(device);

    auto mcts = dz::MCTS{{}};

    int32_t wins = 0;
    int32_t draws = 0;
    int32_t losses = 0;

    auto on_game_end = [&wins, &draws, &losses, &bar_id, &bars](
                            size_t, dz::GameOutcome outcome,
                            dz::Player terminal_player) {
      // outcome from the perspective of the current model
      outcome = terminal_player.is_first() ? outcome : outcome.flip();
      if (outcome == dz::GameOutcome::Win) {
        wins++;
      } else if (outcome == dz::GameOutcome::Draw) {
        draws++;
      } else {
        losses++;
      }

      bars[bar_id].set_option(opt::PostfixText{
          std::format("Evaluating Model | Wins: {} - Draws: {} - Losses: {}",
                      wins, draws, losses)});
      bars[bar_id].tick();
    };

    auto parallel_games =
        az::ParallelGames<dz::Game>(num_evaluation_games, on_game_end);

    while (not parallel_games.all_terminated()) {
      const auto states = parallel_games.get_non_terminal_states();

      const auto action_probs_of_the_current_model =
          mcts.search(states, current_model, num_evaluation_simulations);
      const auto action_probs_of_the_best_model =
          mcts.search(states, best_model, num_evaluation_simulations);

      auto opts = torch::TensorOptions().device(device);
      auto action_probs =
          torch::zeros({static_cast<int32_t>(states.size()), dz::Game::ActionSize}, opts);
      for (const auto i :
            std::views::iota(0, static_cast<int32_t>(states.size()))) {
        if (states[i].player.is_first()) {
          action_probs[i] = action_probs_of_the_current_model[i];
        } else {
          action_probs[i] = action_probs_of_the_best_model[i];
        }
      }
      parallel_games.apply_to_non_terminal_states(action_probs);
    }

    bars[bar_id].mark_as_completed();
  }

  return 0;
}