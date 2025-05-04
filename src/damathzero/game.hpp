#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>

#include "alphazero/game.hpp"
#include "damathzero/board.hpp"

namespace dz {

struct Position {
  const static Position Empty;

  uint8_t x : 3;
  uint8_t y : 3;
  uint8_t empty : 1;

  constexpr Position(int8_t x, int8_t y, bool empty = false)
      : x(x), y(y), empty(empty) {
    assert(x >= 0 and x < 8);
    assert(y >= 0 and y < 8);
  }

  constexpr auto operator==(const Position& other) const -> bool = default;

  constexpr auto is_empty() const -> bool { return empty == 1; }

  constexpr auto value() const -> std::pair<uint8_t, uint8_t> {
    assert(not empty);
    return {x, y};
  }
};
static_assert(sizeof(Position) == 1);

inline constexpr auto Position::Empty = Position{0, 0, true};

struct Game {
  using Action = az::Action;
  using Player = az::Player;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  struct State {
    Board board = Board{};
    std::pair<float16_t, float16_t> scores{0.0, 0.0};
    uint8_t draw_count = 0;
    Player player = Player::First;
    Position eating_piece_position = Position::Empty;
    Position eating_piece_previous_position = Position::Empty;
  };

  struct ActionInfo {
    int8_t distance;
    int8_t direction;

    Position original_position;
    Position new_position;

    bool should_be_knighted;

    float32_t new_score;

    Position eaten_enemy_position = Position::Empty;
  };

  static auto initial_state() -> State;
  static auto legal_actions(const State& state) -> torch::Tensor;
  static auto encode_state(const State& state) -> torch::Tensor;

  static auto decode_action(const State& state, Action action) -> ActionInfo;

  static auto apply_action(const State& state, Action action) -> State;
  static auto inspect_and_apply_action(const State& state, Action action)
      -> std::pair<State, ActionInfo>;

  static auto get_max_eats(const State& state, Action action) -> int32_t;

  static auto get_outcome(const State& state, Action action)
      -> std::optional<az::GameOutcome>;
};

}  // namespace dz

static_assert(az::concepts::Game<dz::Game>);
