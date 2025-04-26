module;

#include <torch/torch.h>

export module dz:game;

import az;
import std;

import :board;

namespace dz {

export struct Game {
  using Action = az::Action;
  using Player = az::Player;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  using Board = Board;

  struct State {
    Board board = Board{};
    Player player = Player::First;
    std::pair<float32_t, float32_t> scores{0.0, 0.0};
    int draw_count = 0;
  };

  struct Position {
    int8_t x;
    int8_t y;
  };

  struct ActionInfo {
    int8_t distance;
    int8_t direction;

    Position original_position;
    Position new_position;

    bool should_be_knighted;

    float32_t new_score;

    std::optional<Position> eaten_enemy_position;
  };

  static constexpr auto initial_state() -> State { return State{}; }

  static auto decode_action(const State& state, Action action) -> ActionInfo {
    const int8_t distance = (action / (8 * 8 * 4)) + 1;
    const int8_t direction = (action % (8 * 8 * 4)) / (8 * 8);
    const int8_t origin_y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    const int8_t origin_x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    auto new_x = origin_x;
    auto new_y = origin_y;

    std::optional<Position> eaten_enemy_position = std::nullopt;

    auto new_score =
        state.player.is_first() ? state.scores.first : state.scores.second;

    auto eat = [&](auto enemy_x, auto enemy_y) {
      assert((state.board[enemy_x, enemy_y].is_occupied));

      auto op = state.board.operators[new_y][new_x];

      auto player_value = state.board[origin_x, origin_y].get_value();
      auto opponent_value = state.board[enemy_x, enemy_y].get_value();

      if (op == '+') {
        new_score += (player_value + opponent_value);
      } else if (op == '-') {
        new_score += (player_value - opponent_value);
      } else if (op == '*') {
        new_score += (player_value * opponent_value);
      } else if (op == '/') {
        new_score += opponent_value > 0 ? (player_value / opponent_value) : 0;
      }

      eaten_enemy_position = Position{enemy_x, enemy_y};
    };

    if (direction == 0) {  // move diagonally to the upper left
      new_x -= distance;
      new_y += distance;

      [&] {
        for (int8_t enemy_y = origin_y + 1; enemy_y < new_y; enemy_y++)
          for (int8_t enemy_x = origin_x - 1; enemy_x > new_x; enemy_x--)
            if (state.board[enemy_x, enemy_y].is_occupied) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 1) {  // move diagonally to the upper right
      new_x += distance;
      new_y += distance;

      [&] {
        for (int8_t enemy_y = origin_y + 1; enemy_y < new_y; enemy_y++)
          for (int8_t enemy_x = origin_x + 1; enemy_x < new_x; enemy_x++)
            if (state.board[enemy_x, enemy_y].is_occupied) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 2) {  // move diagonally to the lower left
      new_x -= distance;
      new_y -= distance;

      [&] {
        for (int8_t enemy_y = origin_y - 1; enemy_y > new_y; enemy_y--)
          for (int8_t enemy_x = origin_x - 1; enemy_x > new_x; enemy_x--)
            if (state.board[enemy_x, enemy_y].is_occupied) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 3) {  // move diagonally to the lower right
      new_x += distance;
      new_y -= distance;

      [&] {
        for (int8_t enemy_y = origin_y - 1; enemy_y > new_y; enemy_y--)
          for (int8_t enemy_x = origin_x + 1; enemy_x < new_x; enemy_x++)
            if (state.board[enemy_x, enemy_y].is_occupied) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    }

    const auto should_be_knighted =
        not state.board[origin_x, origin_y].is_knighted and
        (state.player.is_first() ? new_y == 7 : new_y == 0);

    return {
        .distance = distance,
        .direction = direction,
        .original_position = Position(origin_x, origin_y),
        .new_position = Position(new_x, new_y),
        .should_be_knighted = should_be_knighted,
        .new_score = new_score,
        .eaten_enemy_position = eaten_enemy_position,
    };
  };

  static auto inspect_and_apply_action(const State& state, Action action)
      -> std::pair<State, ActionInfo> {
    const auto action_info = decode_action(state, action);

    const auto [origin_x, origin_y] = action_info.original_position;
    const auto [new_x, new_y] = action_info.new_position;

    auto new_state = state;
    // Move the piece to its new position.
    new_state.board[new_x, new_y] = state.board[origin_x, origin_y];
    new_state.board[origin_x, origin_y] = Board::EmptyCell;

    // If the action resulted to an piece being captured, register it.
    if (auto eaten_enemy_position = action_info.eaten_enemy_position) {
      auto [x, y] = *eaten_enemy_position;
      new_state.board[x, y] = Board::EmptyCell;

      // Update the score if this move triggered an eat.
      if (state.player.is_first()) {
        new_state.scores.first = action_info.new_score;
      } else {
        new_state.scores.second = action_info.new_score;
      }
    }

    new_state.board[new_x, new_y].is_knighted = action_info.should_be_knighted;

    const auto has_eaten = action_info.eaten_enemy_position.has_value();
    const auto can_eat_more =
        has_eaten and
        not new_state.board.get_eatable_actions(new_x, new_y).empty();

    if (can_eat_more) {
      return {new_state, action_info};
    }

    new_state.player = new_state.player.next();

    return {new_state, action_info};
  }

  static auto apply_action(const State& state, Action action) -> State {
    return inspect_and_apply_action(state, action).first;
  }

  static auto get_max_eats(const State& state, Action action) -> int32_t {
    auto stack =
        std::vector<std::tuple<Action, State, int32_t>>{{action, state, 1}};

    auto max_height = 0;
    while (not stack.empty()) {
      auto [action, state, height] = stack.back();
      stack.pop_back();

      max_height = std::max(height, max_height);

      auto [new_state, action_info] = inspect_and_apply_action(state, action);
      auto [new_x, new_y] = action_info.new_position;

      for (auto action : new_state.board.get_eatable_actions(new_x, new_y)) {
        stack.push_back({action, new_state, height + 1});
      }
      // std::println("Here!: {} {}", new_x, new_y);
    }
    return max_height;
  }

  static auto legal_actions(const State& state) -> torch::Tensor {
    // Get positions of the pieces in the board from the perspective of
    // state.player.
    auto positions = std::vector<Position>{};
    for (int8_t y = 0; y < 8; y++) {
      for (int8_t x = 0; x < 8; x++) {
        auto cell = state.board[x, y];
        if (cell.is_occupied and cell.is_owned_by(state.player)) {
          positions.push_back({x, y});
        }
      }
    }

    auto action_and_max_eats = std::vector<std::pair<Action, int32_t>>{};
    for (auto position : positions) {
      for (auto action :
           state.board.get_eatable_actions(position.x, position.y)) {
        action_and_max_eats.push_back({action, get_max_eats(state, action)});
      }
    }

    auto legal_actions = std::vector<Action>{};
    auto best_eats = 0;
    for (auto [action, max_eat] : action_and_max_eats) {
      if (legal_actions.empty()) {
        best_eats = max_eat;
        legal_actions.push_back(action);
        continue;
      }
      if (max_eat < best_eats)
        continue;
      if (max_eat > best_eats) {
        best_eats = max_eat;
        legal_actions.clear();
      }
      legal_actions.push_back(action);
    }

    auto legal_actions_tensor = torch::zeros(ActionSize, torch::kFloat32);

    // NOTE: legal_actions only contains eat moves, so if it's not empty we
    // return early because eat moves are mandatory.
    if (not legal_actions.empty()) {
      for (auto& action : legal_actions)
        legal_actions_tensor[action] = 1.0;

      return legal_actions_tensor;
    }

    for (auto& [x, y] : positions)
      legal_actions.append_range(state.board.get_jump_actions(x, y));

    for (auto& action : legal_actions)
      legal_actions_tensor[action] = 1.0;

    return legal_actions_tensor;
  }

  static constexpr auto get_outcome(const State& state, Action action)
      -> std::optional<az::GameOutcome> {
    if (state.draw_count >= 80)
      return az::GameOutcome::Draw;

    if (legal_actions(state).nonzero().numel() > 0) {
      return {};
    }

    auto distance = (action / (8 * 8 * 4)) + 1;
    auto direction = (action % (8 * 8 * 4)) / (8 * 8);
    auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    auto new_x = x;
    auto new_y = y;

    if (direction == 0) {  // move diagonally to the upper left
      new_x -= distance;
      new_y += distance;
    } else if (direction == 1) {  // move diagonally to the upper right
      new_x += distance;
      new_y += distance;
    } else if (direction == 2) {  // move diagonally to the lower left
      new_x -= distance;
      new_y -= distance;
    } else if (direction == 3) {  // move diagonally to the lower right
      new_x += distance;
      new_y -= distance;
    }

    auto action_played_by_first_player =
        state.board[new_x, new_y].is_owned_by(state.player);

    auto [first, second] = state.scores;
    if (first > second)
      return action_played_by_first_player ? az::GameOutcome::Win
                                           : az::GameOutcome::Loss;
    else if (first < second)
      return action_played_by_first_player ? az::GameOutcome::Loss
                                           : az::GameOutcome::Win;
    else
      return az::GameOutcome::Draw;
  }

  static auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros({8, 8, 6}, torch::kFloat32);
    for (int x = 0; x < 8; x++) {
      for (int y = 0; y < 8; y++) {
        auto [score1, score2] = state.scores;
        if (state.player.is_second()) {
          std::swap(score1, score2);
        }

        encoded_state[x][y][0] = score1;
        encoded_state[x][y][1] = score2;

        if (state.board[x, y].is_occupied) {
          const auto piece = state.board[x, y];
          const auto value = piece.get_value();

          const auto channel_index = (piece.is_owned_by(state.player) ? 2 : 4) +
                                     (piece.is_knighted ? 1 : 0);

          encoded_state[x][y][channel_index] = value;
        }
      }
    }

    return encoded_state;
  }

  static auto print(const State&) -> void {}
};

}  // namespace dz

static_assert(az::concepts::Game<dz::Game>);
