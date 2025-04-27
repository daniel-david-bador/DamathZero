module;

#include <torch/torch.h>

export module dz;

export import :model;
export import :game;
export import :agent;
export import :board;

import az;

namespace dz {

export using Action = az::Action;
export using Config = az::Config;
export using Player = az::Player;
export using GameOutcome = az::GameOutcome;

export using MCTS = az::MCTS<Game, Model>;
export using DamathZero = az::AlphaZero<Game, Model>;

export struct Application {
  DamathZero damathzero;
  MCTS mcts;

  std::shared_ptr<Model> model;

  Game::State state;
  std::optional<GameOutcome> outcome;

  std::vector<Game::State> history;

  std::optional<Action> action_map[8][8][8][8];

  bool destinations[8][8];
  bool moveable_pieces[8][8];

  std::optional<std::pair<int, int>> selected_piece;
  std::vector<std::pair<int, int>> next_moves[8][8];

  torch::Tensor predicted_wdl{};
  torch::Tensor predicted_action_probs{};

  Application()
      : damathzero{dz::Config{
            .num_iterations = 1,
            .num_simulations = 10,
            .num_self_play_iterations_per_actor = 10,
            .num_actors = 5,
            .num_model_evaluation_iterations = 5,
            .num_model_evaluation_simulations = 100,
            .device = torch::kCPU,
        }},
        mcts{dz::Config{
            .num_simulations = 100,
            .num_model_evaluation_iterations = 5,
            .num_model_evaluation_simulations = 100,
            .device = torch::kCPU,
        }},
        model{damathzero.learn({
            .action_size = dz::Game::ActionSize,
            .num_blocks = 2,
            .num_attention_head = 4,
            .embedding_dim = 64,
            .mlp_hidden_size = 128,
            .mlp_dropout_prob = 0.1,
        })},
        state{Game::initial_state()},
        outcome{std::nullopt},
        history{Game::initial_state()} {
    model->to(torch::kCPU);
    update_valid_moves();
  }

  auto update_valid_moves() -> void {
    std::tie(predicted_wdl, predicted_action_probs) =
        model->forward(Game::encode_state(state).unsqueeze(0));

    predicted_wdl = predicted_wdl.squeeze(0);
    predicted_action_probs = predicted_action_probs.squeeze(0).reshape({-1});

    std::memset(action_map, {}, sizeof(action_map));
    std::memset(destinations, {}, sizeof(destinations));
    std::memset(moveable_pieces, {}, sizeof(moveable_pieces));
    std::memset(next_moves, {}, sizeof(next_moves));
    selected_piece = {};

    auto legal_actions = Game::legal_actions(state).nonzero();
    for (auto i = 0; i < legal_actions.size(0); i++) {
      auto action = legal_actions[i].item<int>();

      auto distance = (action / (8 * 8 * 4)) + 1;
      auto direction = (action % (8 * 8 * 4)) / (8 * 8);
      auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
      auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

      moveable_pieces[x][y] = true;

      if (direction == 0) {  // move diagonally to the upper left
        auto new_x = x - distance, new_y = y + distance;
        next_moves[x][y].emplace_back(new_x, new_y);
        action_map[x][y][new_x][new_y] = action;
      } else if (direction == 1) {  // move diagonally to the upper right
        auto new_x = x + distance, new_y = y + distance;
        next_moves[x][y].emplace_back(new_x, new_y);
        action_map[x][y][new_x][new_y] = action;
      } else if (direction == 2) {  // move diagonally to the lower left
        auto new_x = x - distance, new_y = y - distance;
        next_moves[x][y].emplace_back(new_x, new_y);
        action_map[x][y][new_x][new_y] = action;
      } else if (direction == 3) {  // move diagonally to the lower right
        auto new_x = x + distance, new_y = y - distance;
        next_moves[x][y].emplace_back(new_x, new_y);
        action_map[x][y][new_x][new_y] = action;
      }
    }
  }

  auto select_piece(int x, int y) -> void {
    std::memset(destinations, {}, sizeof(destinations));
    selected_piece = {x, y};
    for (auto& [new_x, new_y] : next_moves[x][y])
      destinations[new_x][new_y] = true;
  }

  auto unselect_piece() -> void {
    selected_piece = {};
    std::memset(destinations, {}, sizeof(destinations));
  }

  auto move_piece_to(int new_x, int new_y) -> void {
    auto [x, y] = selected_piece.value();
    auto action = action_map[x][y][new_x][new_y].value();
    state = Game::apply_action(state, action);
    outcome = Game::get_outcome(state, action);

    while (not outcome and state.player.is_second()) {
      auto probs = mcts.search(state, model, 100);
      auto action = torch::argmax(probs).item<Action>();
      state = Game::apply_action(state, action);
      outcome = Game::get_outcome(state, action);
    }

    history.push_back(state);
    update_valid_moves();
  }

  auto undo_move() -> void {
    if (history.size() > 1) {
      history.pop_back();
      state = history.back();
      outcome = std::nullopt;
      update_valid_moves();
    }
  }

  auto reset_game() -> void {
    state = Game::initial_state();
    outcome = std::nullopt;
    update_valid_moves();
  }
};

}  // namespace dz
