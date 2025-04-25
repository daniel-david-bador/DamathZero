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

      eaten_enemy_position = Position{enemy_y, enemy_y};
    };

    if (direction == 0) {  // move diagonally to the upper left
      new_x -= distance;
      new_y += distance;

      [&] {
        for (int8_t enemy_y = origin_y + 1; enemy_y < new_y; enemy_y++)
          for (int8_t enemy_x = origin_x - 1; enemy_x > new_x; enemy_x--)
            if (state.board[enemy_x, enemy_y].occup) {
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
            if (state.board[enemy_x, enemy_y].occup) {
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
            if (state.board[enemy_x, enemy_y].occup) {
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
            if (state.board[enemy_x, enemy_y].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    }

    const auto should_be_knighted =
        not state.board[origin_x, origin_y].queen and new_y == 7;

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
    auto new_state = state;

    auto action_info = decode_action(state, action);

    auto [origin_x, origin_y] = action_info.original_position;
    auto [new_x, new_y] = action_info.new_position;

    new_state.board[new_x, new_y] = state.board[origin_x, origin_y];
    new_state.board[origin_x, origin_y] = Board::EmptyCell;

    if (auto eaten_enemy_position = action_info.eaten_enemy_position) {
      auto [x, y] = *eaten_enemy_position;
      new_state.board[x, y] = Board::EmptyCell;
    }

    auto can_eat_more =
        not new_state.board.get_eatable_actions(new_x, new_y).empty();
    if (not action_info.should_be_knighted and can_eat_more) {
      return {new_state, action_info};
    }

    if (action_info.should_be_knighted) {
      new_state.board[new_x, new_y].queen = 1;
    }

    new_state.board = new_state.board.flip();
    new_state.player = state.player.next();

    if (state.player.is_first()) {
      new_state.scores.first = action_info.new_score;
    } else {
      new_state.scores.second = action_info.new_score;
    }

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
        if (cell.occup and not cell.enemy) {
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

    if (legal_actions(state).sum(0).item<float64_t>() == 0.0) {
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

      auto canonical_state = state;
      canonical_state.board = state.board.flip();
      canonical_state.player = state.player.next();
      canonical_state.scores.first = state.scores.second;
      canonical_state.scores.second = state.scores.first;

      auto piece = canonical_state.board[new_x, new_y];

      auto [first, second] = canonical_state.scores;
      if (first > second)
        return not piece.enemy ? az::GameOutcome::Win : az::GameOutcome::Loss;
      else if (first < second)
        return not piece.enemy ? az::GameOutcome::Loss : az::GameOutcome::Win;
      else
        return az::GameOutcome::Draw;
    } else
      return {};
  }

  static auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros({8, 8, 6}, torch::kFloat32);
    for (int x = 0; x < 8; x++) {
      for (int y = 0; y < 8; y++) {
        if (state.player.is_first()) {
          encoded_state[x][y][0] = state.scores.first;
          encoded_state[x][y][1] = state.scores.second;
          if (state.board.cells[y][x].occup) {
            auto& piece = state.board.cells[y][x];
            auto value = (piece.ngtve ? -1 : 1) * piece.value;
            encoded_state[x][y][(not piece.enemy ? 2 : 4) +
                                (piece.queen ? 1 : 0)] = value;
          }
        } else {
          encoded_state[x][y][0] = state.scores.second;
          encoded_state[x][y][1] = state.scores.first;
          if (state.board.cells[y][x].occup) {
            auto& piece = state.board.cells[y][x];
            auto value = (piece.ngtve ? -1 : 1) * piece.value;
            encoded_state[x][y][(piece.enemy ? 2 : 4) + (piece.queen ? 1 : 0)] =
                value;
          }
        }
      }
    }

    return encoded_state;
  }

  static auto print(const State& state) -> void {
    auto& board = state.board;
    auto [first, second] = state.scores;
    std::print("Score: {:.5f} {:.5f}\n", first, second);
    for (int i = 7; i >= 0; i--) {
      std::print(" {: ^3} ", i);
      for (int j = 0; j < 8; j++) {
        if (board.cells[i][j].occup) {
          if (board.cells[i][j].queen)
            std::cout << "\033[1m";  // bold
          std::cout << (board.cells[i][j].enemy ? "\033[31m"
                                                : "\033[34m");  // blue
          std::print(" {: ^3} ", (board.cells[i][j].ngtve ? -1 : 1) *
                                     board.cells[i][j].value);
          std::cout << "\033[0m";  // reset color
        } else {
          std::print(" {: ^3} ", board.operators[i][j]);
        }
      }
      std::println();
    }

    std::print("     ");
    for (int i = 0; i < 8; i++) {
      std::print(" {: ^3} ", i);
    }
    std::println();
  }
};

};  // namespace dz

static_assert(az::concepts::Game<dz::Game>);
