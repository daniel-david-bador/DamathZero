#include <torch/torch.h>

#include "alphazero/az.hpp"
#include "damathzero/game.hpp"
#include "damathzero/model.hpp"

namespace dz {
using Action = az::Action;
using Player = az::Player;
using GameOutcome = az::GameOutcome;

using MCTS = az::MCTS<Game, Model>;
using DamathZero = az::AlphaZero<Game, Model>;

using DeviceType = torch::DeviceType;

auto save_model(std::shared_ptr<Model> model, std::string_view path) -> void;

auto load_model(std::string_view path, Model::Config config)
    -> std::shared_ptr<Model>;

struct Application {
  struct Config {
    int32_t num_simulations = 1000;
  };

  Application(Config config, Model::Config model_config, std::string_view path,
              Game::State initial_state = Game::initial_state());

  auto reset_valid_moves() -> void;
  auto update_valid_moves() -> void;
  auto select_piece(int x, int y) -> void;
  auto unselect_piece() -> void;
  auto let_ai_move() -> void;
  auto move_piece_to(int new_x, int new_y) -> void;
  auto update_final_scores() -> void;
  auto undo_move() -> void;
  auto reset_game() -> void;
  auto wdl_probs() const -> std::optional<std::array<float, 3>>;
  auto action_probs(int i, int j) const -> float;
  auto max_action_probs(int i, int j) const -> float;

  MCTS mcts;
  Config config;

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
