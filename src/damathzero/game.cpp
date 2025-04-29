module;

#include <torch/torch.h>

export module dz:game;

import :board;

import az;
import std;

namespace dz {

export struct Game {
  using Action = az::Action;
  using Player = az::Player;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  using Board = Board;

  struct Position {
    int8_t x;
    int8_t y;
  };

  struct State {
    Board board = Board{};
    Player player = Player::First;
    std::pair<float32_t, float32_t> scores{0.0, 0.0};
    int draw_count = 0;
    std::optional<Position> last_piece_moved = std::nullopt;
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

  static constexpr auto initial_state() -> State {
    return State{
        .player = std::mt19937{std::random_device{}()}() % 2 == 0
                      ? Player::First
                      : Player::Second,
    };
  }

  static auto decode_action(const State& state, Action action) -> ActionInfo {
    const int8_t distance = (action / (8 * 8 * 4)) + 1;
    const int8_t direction = (action % (8 * 8 * 4)) / (8 * 8);

    const int8_t origin_y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    const int8_t origin_x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    const auto [dx, dy] = Board::directions[direction];

    const int8_t new_x = origin_x + dx * distance;
    const int8_t new_y = origin_y + dy * distance;

    std::optional<Position> eaten_enemy_position = std::nullopt;

    auto new_score =
        state.player.is_first() ? state.scores.first : state.scores.second;

    for (auto enemy_distance = distance - 1; enemy_distance > 0;
         enemy_distance--) {
      const int8_t enemy_x = origin_x + dx * enemy_distance;
      const int8_t enemy_y = origin_y + dy * enemy_distance;

      if (not state.board[enemy_x, enemy_y].is_occupied)
        continue;

      auto op = state.board.operators[new_y][new_x];

      auto player_value = state.board[origin_x, origin_y].value();
      auto opponent_value = state.board[enemy_x, enemy_y].value();

      auto multiplier = 1.0;

      if (state.board[origin_x, origin_y].is_knighted or
          state.board[enemy_x, enemy_y].is_knighted) {
        multiplier = 2.0;
      }

      if (state.board[origin_x, origin_y].is_knighted and
          state.board[enemy_x, enemy_y].is_knighted) {
        multiplier = 4.0;
      }

      if (op == '+') {
        new_score += (player_value + opponent_value) * multiplier;
      } else if (op == '-') {
        new_score += (player_value - opponent_value) * multiplier;
      } else if (op == '*') {
        new_score += (player_value * opponent_value) * multiplier;
      } else if (op == '/' and opponent_value != 0) {
        new_score += (player_value / opponent_value) * multiplier;
      }

      eaten_enemy_position = Position{enemy_x, enemy_y};
      break;
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
    new_state.draw_count += 1;

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

      new_state.draw_count = 0;
    }

    if (action_info.should_be_knighted)
      new_state.board[new_x, new_y].is_knighted = true;

    const auto has_eaten = action_info.eaten_enemy_position.has_value();
    const auto can_eat_more =
        has_eaten and not action_info.should_be_knighted and
        not new_state.board.get_eatable_actions(new_x, new_y).empty();

    if (can_eat_more) {
      new_state.last_piece_moved = action_info.new_position;
      return {new_state, action_info};
    }

    new_state.player = new_state.player.next();
    new_state.last_piece_moved = std::nullopt;
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
      if (new_state.last_piece_moved) {
        auto [new_x, new_y] = action_info.new_position;

        for (auto action : new_state.board.get_eatable_actions(new_x, new_y)) {
          stack.push_back({action, new_state, height + 1});
        }
      }
    }
    return max_height;
  }

  static auto legal_actions(const State& state) -> torch::Tensor {
    // If last piece moved is not empty, get positions of the pieces in the
    // board from the perspective of state.player.
    auto positions = std::vector<Position>{};
    if (state.last_piece_moved) {
      positions.push_back(*state.last_piece_moved);
    } else {
      for (int8_t y = 0; y < 8; y++) {
        for (int8_t x = 0; x < 8; x++) {
          auto cell = state.board[x, y];
          if (cell.is_occupied and cell.is_owned_by(state.player)) {
            positions.push_back({x, y});
          }
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

    for (auto [x, y] : positions)
      legal_actions.append_range(state.board.get_jump_actions(x, y));

    for (auto action : legal_actions)
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

    const int8_t distance = (action / (8 * 8 * 4)) + 1;
    const int8_t direction = (action % (8 * 8 * 4)) / (8 * 8);

    const int8_t origin_y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    const int8_t origin_x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    auto [dx, dy] = Board::directions[direction];

    const int8_t new_x = origin_x + dx * distance;
    const int8_t new_y = origin_y + dy * distance;

    const auto action_played_by_first_player =
        state.board[new_x, new_y].is_owned_by_first_player;

    auto [first_player_score, second_player_score] = state.scores;

    for (const auto row : state.board.cells) {
      for (const auto cell : row) {
        if (cell.is_occupied) {
          const auto cell_value = cell.value() * (cell.is_knighted ? 2 : 1);
          cell.is_owned_by_first_player ? first_player_score += cell_value
                                        : second_player_score += cell_value;
        }
      }
    }

    if (first_player_score > second_player_score)
      return action_played_by_first_player ? az::GameOutcome::Win
                                           : az::GameOutcome::Loss;
    else if (first_player_score < second_player_score)
      return action_played_by_first_player ? az::GameOutcome::Loss
                                           : az::GameOutcome::Win;
    else
      return az::GameOutcome::Draw;
  }

  static auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros({8, 8, 10}, torch::kFloat32);

    encoded_state.select(2, 0).fill_(state.player.is_first() ? 0.0 : 1.0);
    encoded_state.select(2, 1).fill_(std::tanh(
        state.player.is_first() ? (state.scores.first - state.scores.second)
                                : (state.scores.second - state.scores.first)));
    encoded_state.select(2, 2).fill_(state.draw_count / 80.0);

    for (int x = 0; x < 8; x++) {
      for (int y = 0; y < 8; y++) {
        switch (state.board.operators[y][x]) {
          case '+':
            encoded_state[x][y][3] = 1.0;
            break;
          case '-':
            encoded_state[x][y][3] = 2.0;
            break;
          case '*':
            encoded_state[x][y][3] = 3.0;
            break;
          case '/':
            encoded_state[x][y][3] = 4.0;
            break;
          default:
            break;
        }

        if (state.board[x, y].is_occupied) {
          const auto piece = state.board[x, y];
          const auto value = piece.value();

          if (piece.is_owned_by(state.player))
            encoded_state[x][y][piece.is_knighted ? 5 : 4] = value;
          else
            encoded_state[x][y][piece.is_knighted ? 7 : 6] = value;
        }
      }
    }

    if (state.last_piece_moved) {
      const auto [x, y] = *state.last_piece_moved;

      encoded_state.select(2, 8).fill_(1.0);
      encoded_state.select(2, 9).fill_(state.board[x, y].value());
    }

    return encoded_state;
  }

  static auto print(const State&) -> void {}
};

}  // namespace dz

static_assert(az::concepts::Game<dz::Game>);
