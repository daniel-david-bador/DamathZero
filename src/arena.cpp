module;

#include <torch/torch.h>

export module alphazero:arena;

import std;
import :model;
import :game;
import :mcts;
import :config;

namespace AZ {

namespace Concepts {

export template <typename C, typename G>
concept Agent =
    Concepts::Game<G> and requires(C c, G::State state, torch::Tensor probs,
                                   GameOutcome outcome, Action action) {
      { c.on_move(state) } -> std::same_as<Action>;
      { c.on_model_move(state, probs, action) } -> std::same_as<void>;
      { c.on_game_end(state, outcome) } -> std::same_as<void>;
    };

}  // namespace Concepts

export template <Concepts::Game Game, Concepts::Model Model>
class Arena {
  using State = Game::State;

 public:
  struct Result {
    int wins;
    int draws;
    int losses;
  };

  Arena(Config config) : config_(config) {}

  auto play_with_model(std::shared_ptr<Model> model, int num_model_simulations,
                       Concepts::Agent<Game> auto controller,
                       Player human_player) -> void {
    auto mcts = MCTS<Game, Model>(config_);

    model->eval();
    auto state = Game::initial_state();

    auto action = -1;
    while (true) {
      if (state.player == human_player) {
        action = controller.on_move(state);
      } else {
        auto probs = mcts.search(state, model, num_model_simulations);
        action = torch::argmax(probs).template item<Action>();
        controller.on_model_move(state, probs, action);
      }

      auto new_state = Game::apply_action(state, action);
      auto outcome = Game::get_outcome(new_state, action);

      if (outcome) {
        controller.on_game_end(new_state, human_player == state.player
                                              ? *outcome
                                              : outcome->flip());
        break;
      }

      state = std::move(new_state);
    }
  }

  auto play(std::shared_ptr<Model> model1, std::shared_ptr<Model> model2,
            int rounds, int num_simulations) -> Result {
    model1->to(config_.device);
    model2->to(config_.device);
    model1->eval();
    model2->eval();

    auto wins = 0;
    auto draws = 0;
    auto losses = 0;

    auto mcts = MCTS<Game, Model>(config_);

    for (auto _ : std::views::iota(0, rounds)) {
      auto state = Game::initial_state();

      while (true) {
        auto model = state.player.is_first() ? model1 : model2;
        auto action_probs = mcts.search(state, model, num_simulations);

        auto action = torch::argmax(action_probs).template item<Action>();

        auto new_state = Game::apply_action(state, action);

        if (auto outcome = Game::get_outcome(new_state, action)) {
          // outcome from the perspective of model1
          auto flipped_outcome =
              state.player.is_first() ? *outcome : outcome->flip();

          if (flipped_outcome == GameOutcome::Win) {
            wins += 1;
          } else if (flipped_outcome == GameOutcome::Draw) {
            draws += 1;
          } else if (flipped_outcome == GameOutcome::Loss) {
            losses += 1;
          }

          break;
        }

        state = std::move(new_state);
      }
    }

    return {wins, draws, losses};
  }

 private:
  Config config_;
};

};  // namespace AZ
