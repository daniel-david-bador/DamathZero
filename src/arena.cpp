module;

#include <torch/torch.h>

export module alphazero:arena;

import std;
import :game;
import :mcts;
import :config;

namespace AlphaZero {

export struct Result {
  int wins;
  int draws;
  int losses;
};

export enum GameResult {
  Win,
  Lost,
  Draw,
};

namespace Concepts {

export template <typename C, typename G>
concept Agent =
    Concepts::Game<G> and requires(C c, G::State state, torch::Tensor probs,
                                   GameResult result, Action action) {
      std::same_as<decltype(C::player), Player>;
      { c.on_move(state) } -> std::same_as<Action>;
      { c.on_model_move(state, probs, action) } -> std::same_as<void>;
      { c.on_game_end(state, result) } -> std::same_as<void>;
    };

}  // namespace Concepts

export template <Concepts::Game Game>
class Arena {
  using Network = Game::Network;
  using State = Game::State;

 public:
  Arena(Config config) : config_(config) {}

  auto play_with_model(std::shared_ptr<Network> model,
                       int num_model_simulations,
                       Concepts::Agent<Game> auto controller) -> void {
    auto mcts = MCTS<Game>(config_);

    model->eval();
    auto state = Game::initial_state();

    auto action = -1;
    while (true) {
      if (state.player == decltype(controller)::player) {
        action = controller.on_move(state);
      } else {
        auto probs = mcts.search(state, model, num_model_simulations);
        action = torch::argmax(probs).template item<Action>();
        controller.on_model_move(state, probs, action);
      }

      auto new_state = Game::apply_action(state, action);
      auto terminal_value = Game::terminal_value(new_state, action);

      if (terminal_value.has_value()) {
        auto value = controller.player == state.player ? *terminal_value
                                                       : -*terminal_value;

        controller.on_game_end(new_state, value == 0   ? GameResult::Draw
                                          : value == 1 ? GameResult::Win
                                                       : GameResult::Lost);
        break;
      }

      state = new_state;
    }
  }

  auto play(std::shared_ptr<Network> model1, std::shared_ptr<Network> model2,
            int rounds, int num_simulations) -> Result {
    model1->eval();
    model2->eval();

    auto wins = 0;
    auto draws = 0;
    auto losses = 0;

    auto mcts = MCTS<Game>(config_);

    for (auto _ : std::views::iota(0, rounds)) {
      auto state = Game::initial_state();
      auto value = 0.0;

      while (true) {
        auto model = state.player.is_first() ? model1 : model2;
        auto action_probs = mcts.search(state, model, num_simulations);

        auto action = torch::argmax(action_probs).template item<Action>();

        auto new_state = Game::apply_action(state, action);
        auto terminal_value = Game::terminal_value(new_state, action);

        if (terminal_value.has_value()) {
          if (value == 0) {
            draws += 1;
          } else if (value == 1) {
            if (state.player.is_first()) {
              wins += 1;
            } else {
              losses += 1;
            }
          }
          break;
        }

        state = new_state;
      }
    }

    return {wins, draws, losses};
  }

 private:
  Config config_;
};

};  // namespace AlphaZero
