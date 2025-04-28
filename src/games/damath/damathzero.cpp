module;

#include <torch/torch.h>

export module dz;

export import :model;
export import :game;
export import :board;

import az;
import std;

namespace dz {

export using Config = az::Config;

export using Action = az::Action;
export using Player = az::Player;
export using GameOutcome = az::GameOutcome;

export using MCTS = az::MCTS<Game, Model>;
export using DamathZero = az::AlphaZero<Game, Model>;

export using DeviceType = at::DeviceType;

export auto save_model(std::shared_ptr<Model> model, std::string_view path)
    -> void {
  az::utils::save_model(model, path);
}

export auto load_model(std::string_view path, Model::Config config)
    -> std::shared_ptr<Model> {
  return az::utils::load_model<Model>(path, config);
}

export struct Application {
  Application(Config config, Model::Config model_config, std::string_view path)
      : config{config},
        damathzero{config},
        mcts{config},
        model{load_model(path, model_config)},
        state{Game::initial_state()},
        outcome{std::nullopt},
        history{Game::initial_state()} {
    update_valid_moves();
  }

  auto reset_valid_moves() -> void {
    std::memset(action_map, {}, sizeof(action_map));
    std::memset(destinations, {}, sizeof(destinations));
    std::memset(moveable_pieces, {}, sizeof(moveable_pieces));
    std::memset(next_moves, {}, sizeof(next_moves));

    selected_piece = std::nullopt;
    predicted_wdl = std::nullopt;
    predicted_action_probs = std::nullopt;
  }

  auto update_valid_moves() -> void {
    reset_valid_moves();

    auto legal_actions = Game::legal_actions(state).nonzero();
    for (auto i = 0; i < legal_actions.size(0); i++) {
      auto action = legal_actions[i].item<int>();
      auto action_info = Game::decode_action(state, action);

      auto [origin_x, origin_y] = action_info.original_position;
      auto [new_x, new_y] = action_info.new_position;

      moveable_pieces[origin_x][origin_y] = true;

      next_moves[origin_x][origin_y].emplace_back(new_x, new_y);
      action_map[origin_x][origin_y][new_x][new_y] = action;
    }

    model->eval();
    std::tie(predicted_wdl, predicted_action_probs) =
        model->forward(Game::encode_state(state).unsqueeze(0));

    predicted_wdl = predicted_wdl.value().squeeze(0);
    predicted_action_probs =
        predicted_action_probs.value().squeeze(0).reshape({-1});
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

  auto let_ai_move() -> void {
    auto probs = mcts.search(state, model, config.num_simulations);
    auto action = torch::argmax(probs).item<Action>();
    state = Game::apply_action(state, action);
    outcome = Game::get_outcome(state, action);

    if (outcome.has_value())
      update_final_scores();
    else
      update_valid_moves();

    history.push_back(state);
  }

  auto move_piece_to(int new_x, int new_y) -> void {
    auto [x, y] = selected_piece.value();
    auto action = action_map[x][y][new_x][new_y].value();
    state = Game::apply_action(state, action);
    outcome = Game::get_outcome(state, action);

    if (outcome.has_value())
      update_final_scores();
    else
      update_valid_moves();

    history.push_back(state);
  }

  auto update_final_scores() -> void {
    reset_valid_moves();
    auto& [first_player_score, second_player_score] = state.scores;

    for (const auto row : state.board.cells) {
      for (const auto cell : row) {
        if (cell.is_occupied) {
          const auto cell_value = cell.value() * (cell.is_knighted ? 2 : 1);
          cell.is_owned_by_first_player ? first_player_score += cell_value
                                        : second_player_score += cell_value;
        }
      }
    }
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

    history.push_back(state);
    update_valid_moves();
  }

  auto wdl_probs() const -> std::optional<std::array<float, 3>> {
    if (not predicted_wdl.has_value())
      return std::nullopt;

    auto wdl = predicted_wdl.value();
    std::array<float, 3> result;
    if (state.player.is_first()) {
      result[0] = wdl[0].item<float>();
      result[1] = wdl[1].item<float>();
      result[2] = wdl[2].item<float>();
    } else {
      result[0] = wdl[2].item<float>();
      result[1] = wdl[1].item<float>();
      result[2] = wdl[0].item<float>();
    }

    return result;
  }

  auto action_probs(int i, int j) const -> float {
    if (not predicted_action_probs.has_value() or
        not selected_piece.has_value())
      return 0.0f;

    auto [x, y] = selected_piece.value();
    auto action = action_map[x][y][i][j].value();
    auto action_probs = predicted_action_probs.value()[action].item<float>();
    return action_probs;
  }

  auto max_action_probs(int i, int j) const -> float {
    if (not predicted_action_probs.has_value())
      return 0.0f;

    auto actions = action_map[i][j];
    auto max_action_probs = 0.0f;
    for (auto k = 0; k < 8; k++)
      for (auto l = 0; l < 8; l++)
        if (actions[k][l].has_value()) {
          auto action = actions[k][l].value();
          auto action_probs =
              predicted_action_probs.value()[action].item<float>();
          if (std::abs(action_probs) > std::abs(max_action_probs))
            max_action_probs = action_probs;
        }

    return max_action_probs;
  }

  Config config;

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

  std::optional<torch::Tensor> predicted_wdl{};
  std::optional<torch::Tensor> predicted_action_probs{};
};

}  // namespace dz
