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
    std::optional<Position> eating_piece_position = std::nullopt;
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
      new_state.eating_piece_position = action_info.new_position;
      return {new_state, action_info};
    }

    new_state.player = new_state.player.next();
    new_state.eating_piece_position = std::nullopt;
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
      if (new_state.eating_piece_position) {
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
    if (state.eating_piece_position) {
      positions.push_back(*state.eating_piece_position);
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

    auto eat_actions = std::vector<std::tuple<Position, Action, int32_t>>{};
    for (auto position : positions) {
      for (auto action :
           state.board.get_eatable_actions(position.x, position.y)) {
        eat_actions.push_back({position, action, get_max_eats(state, action)});
      }
    }

    auto dama_actions = std::vector<Action>{};
    auto normal_actions = std::vector<Action>{};

    auto best_eats = 0;
    for (auto [position, action, max_eat] : eat_actions) {
      if (max_eat < best_eats)
        continue;

      if (max_eat > best_eats) {
        best_eats = max_eat;
        dama_actions.clear();
        normal_actions.clear();
      }

      if (state.board[position.x, position.y].is_knighted)
        dama_actions.push_back(action);
      else
        normal_actions.push_back(action);
    }

    auto legal_actions = torch::zeros(ActionSize, torch::kFloat32);

    for (auto& action : dama_actions)
      legal_actions[action] = 1.0;

    if (not dama_actions.empty())
      return legal_actions;

    for (auto& action : normal_actions)
      legal_actions[action] = 1.0;

    if (not normal_actions.empty())
      return legal_actions;

    auto jump_actions = std::vector<Action>{};

    for (auto [x, y] : positions)
      jump_actions.append_range(state.board.get_jump_actions(x, y));

    for (auto action : jump_actions)
      legal_actions[action] = 1.0;

    return legal_actions;
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
    auto encoded_state = torch::zeros({32, 11}, torch::kFloat32);
    constexpr auto operator_index = [](int8_t x, int8_t y) {
      switch (Board::operators[y][x]) {  // clang-format off
        case '+': return 0;
        case '-': return 1;
        case '*': return 2;
        case '/': return 3;
        default: std::unreachable();
      }  // clang-format on
    };

    const auto [score1, score2] = state.scores;
    auto i = 0;
    for (int8_t y = 0; y < 8; y += 1) {
      for (int8_t x = y % 2 == 0 ? 1 : 0; x < 8; x += 2) {
        auto cell = state.board[x, y];
        if (const auto piece = cell; cell.is_occupied) {
          // 0-11 is the one-hot encoding of each piece
          encoded_state[i][0] = piece.value();
          // 12 encodes the owner of the piece
          encoded_state[i][1] = piece.is_owned_by_first_player ? 1 : 0;
          // 13 encodes if the current player "owns" this piece
          encoded_state[i][2] = piece.is_owned_by(state.player) ? 1 : 0;
          // 14 encodes if this piece is promoted
          encoded_state[i][3] = piece.is_knighted ? 1 : 0;

          const auto score = piece.is_owned_by_first_player ? score1 - score2
                                                            : score2 - score1;
          const auto relative_score = 1 / (1 + std::exp(-score));

          // 15 encodes the relative score of the piece
          encoded_state[i][4] = relative_score;
        }

        encoded_state[i][5] = state.draw_count / 80.0;

        // 16-19 is the one-hot encoding of each operator
        encoded_state[i][6 + operator_index(x, y)] = 1.0;
        i++;
      }
    }

    if (const auto pos = state.eating_piece_position) {
      const auto [x, y] = *pos;
      const auto i = (4 * y) + (x / 2);
      encoded_state[i][10] = 1;
    }

    return encoded_state;
  }

  static auto print(const State&) -> void {}
};

}  // namespace dz

static_assert(az::concepts::Game<dz::Game>);
