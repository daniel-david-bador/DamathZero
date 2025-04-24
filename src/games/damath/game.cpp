module;

#include <torch/torch.h>

export module dz:game;

import az;
import std;

namespace dz {

export struct Game {
  using Action = az::Action;
  using Player = az::Player;

  static constexpr auto ActionSize = 8 * 8 * 4 * 7;

  struct Board {
    struct Piece {
      std::uint8_t occup : 1;
      std::uint8_t enemy : 1;
      std::uint8_t queen : 1;
      std::uint8_t ngtve : 1;
      std::uint8_t value : 4;
    };

    static constexpr std::array<std::array<const char, 8>, 8> operators{
        {{' ', '+', ' ', '-', ' ', '/', ' ', '*'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'*', ' ', '/', ' ', '-', ' ', '+', ' '},
         {' ', '+', ' ', '-', ' ', '/', ' ', '*'},
         {'-', ' ', '+', ' ', '*', ' ', '/', ' '},
         {' ', '/', ' ', '*', ' ', '+', ' ', '-'},
         {'*', ' ', '/', ' ', '-', ' ', '+', ' '}}};

    std::array<std::array<Board::Piece, 8>, 8> pieces{{
        // clang-format off
        {{{0,0,0,0,0},{1,0,0,1,11},{0,0,0,0,0},{1,0,0,0,8},{0,0,0,0,0},{1,0,0,1,5},{0,0,0,0,0},{1,0,0,0,2}}},
        {{{1,0,0,0,0},{0,0,0,0,0},{1,0,0,1,3},{0,0,0,0,0},{1,0,0,0,10},{0,0,0,0,0},{1,0,0,1,7},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,0,0,1,9},{0,0,0,0,0},{1,0,0,0,6},{0,0,0,0,0},{1,0,0,1,1},{0,0,0,0,0},{1,0,0,0,4}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}},
        {{{1,1,0,0,4},{0,0,0,0,0},{1,1,0,1,1},{0,0,0,0,0},{1,1,0,0,6},{0,0,0,0,0},{1,1,0,1,9},{0,0,0,0,0}}},
        {{{0,0,0,0,0},{1,1,0,1,7},{0,0,0,0,0},{1,1,0,0,10},{0,0,0,0,0},{1,1,0,1,3},{0,0,0,0,0},{1,1,0,0,0}}},
        {{{1,1,0,0,2},{0,0,0,0,0},{1,1,0,1,5},{0,0,0,0,0},{1,1,0,0,8},{0,0,0,0,0},{1,1,0,1,11},{0,0,0,0,0}}},
      }};  // clang-format on
  };

  struct State {
    Board board = Board{};
    Player player = Player::First;
    std::pair<double, double> scores{0.0, 0.0};
    int draw_count = 0;
  };

  static constexpr auto flip(const Board& board) -> Board {
    auto new_board = Board{
        .pieces =  // clang-format off
              {{{{board.pieces[7][7], board.pieces[7][6], board.pieces[7][5], board.pieces[7][4], board.pieces[7][3], board.pieces[7][2], board.pieces[7][1], board.pieces[7][0]}},
                {{board.pieces[6][7], board.pieces[6][6], board.pieces[6][5], board.pieces[6][4], board.pieces[6][3], board.pieces[6][2], board.pieces[6][1], board.pieces[6][0]}},
                {{board.pieces[5][7], board.pieces[5][6], board.pieces[5][5], board.pieces[5][4], board.pieces[5][3], board.pieces[5][2], board.pieces[5][1], board.pieces[5][0]}},
                {{board.pieces[4][7], board.pieces[4][6], board.pieces[4][5], board.pieces[4][4], board.pieces[4][3], board.pieces[4][2], board.pieces[4][1], board.pieces[4][0]}},
                {{board.pieces[3][7], board.pieces[3][6], board.pieces[3][5], board.pieces[3][4], board.pieces[3][3], board.pieces[3][2], board.pieces[3][1], board.pieces[3][0]}},
                {{board.pieces[2][7], board.pieces[2][6], board.pieces[2][5], board.pieces[2][4], board.pieces[2][3], board.pieces[2][2], board.pieces[2][1], board.pieces[2][0]}},
                {{board.pieces[1][7], board.pieces[1][6], board.pieces[1][5], board.pieces[1][4], board.pieces[1][3], board.pieces[1][2], board.pieces[1][1], board.pieces[1][0]}},
                {{board.pieces[0][7], board.pieces[0][6], board.pieces[0][5], board.pieces[0][4], board.pieces[0][3], board.pieces[0][2], board.pieces[0][1], board.pieces[0][0]}}
              }},  // clang-format on
    };

    for (auto& row : new_board.pieces)
      for (auto& piece : row)
        if (piece.occup)
          piece.enemy = not piece.enemy;

    return new_board;
  }

  static constexpr auto initial_state() -> State { return State{}; }

  static constexpr auto apply_action(const State& state, Action action)
      -> State {
    auto distance = (action / (8 * 8 * 4)) + 1;
    auto direction = (action % (8 * 8 * 4)) / (8 * 8);
    auto y = ((action % (8 * 8 * 4)) % (8 * 8)) / 8;
    auto x = ((action % (8 * 8 * 4)) % (8 * 8)) % 8;

    auto new_state = state;
    auto new_x = x;
    auto new_y = y;

    new_state.draw_count += 1;
    auto has_eaten = false;

    auto eat = [&](auto enemy_x, auto enemy_y) {
      auto op = state.board.operators[new_y][new_x];
      auto& score = new_state.scores.first;
      auto player_value =
          static_cast<double>((state.board.pieces[y][x].ngtve ? -1 : 1) *
                              state.board.pieces[y][x].value);
      auto opponent_value = static_cast<double>(
          (state.board.pieces[enemy_y][enemy_x].ngtve ? -1 : 1) *
          state.board.pieces[enemy_y][enemy_x].value);
      if (op == '+') {
        score += (player_value + opponent_value);
      } else if (op == '-') {
        score += (player_value - opponent_value);
      } else if (op == '*') {
        score += (player_value * opponent_value);
      } else if (op == '/') {
        score += opponent_value > 0 ? (player_value / opponent_value) : 0;
      }
      new_state.draw_count = 0;
      new_state.board.pieces[enemy_y][enemy_x] = {0, 0, 0, 0, 0};
      has_eaten = true;
    };

    if (direction == 0) {  // move diagonally to the upper left
      new_x -= distance;
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 1) {  // move diagonally to the upper right
      new_x += distance;
      new_y += distance;

      [&] {
        for (int enemy_y = y + 1; enemy_y < new_y; enemy_y++)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 2) {  // move diagonally to the lower left
      new_x -= distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x - 1; enemy_x > new_x; enemy_x--)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    } else if (direction == 3) {  // move diagonally to the lower right
      new_x += distance;
      new_y -= distance;

      [&] {
        for (int enemy_y = y - 1; enemy_y > new_y; enemy_y--)
          for (int enemy_x = x + 1; enemy_x < new_x; enemy_x++)
            if (state.board.pieces[enemy_y][enemy_x].occup) {
              eat(enemy_x, enemy_y);
              return;
            }
      }();
    }

    new_state.board.pieces[y][x] = {0, 0, 0, 0, 0};
    new_state.board.pieces[new_y][new_x] = state.board.pieces[y][x];

    auto is_knighted = false;
    if ((not new_state.board.pieces[new_y][new_x].queen) and (new_y == 7)) {
      new_state.board.pieces[new_y][new_x].queen = true;
      is_knighted = true;
    }

    if ((not is_knighted) and has_eaten and
        get_eat_actions(new_state, new_x, new_y).size() > 0)
      return new_state;

    auto [score1, score2] = new_state.scores;

    new_state.board = flip(new_state.board);
    new_state.player = state.player.next();
    new_state.scores.first = score2;
    new_state.scores.second = score1;
    return new_state;
  }

  struct ActionNode {
    int action;
    std::vector<ActionNode> children;

    auto height() const -> int {
      if (children.empty())
        return 1;
      else
        return 1 + std::ranges::max(
                       children | std::views::transform([](const auto& child) {
                         return child.height();
                       }));
    }
  };

  static constexpr auto get_eat_actions(const State& state, int x, int y)
      -> std::vector<ActionNode> {
    auto actions = std::vector<ActionNode>{};

    auto piece = state.board.pieces[y][x];

    if (piece.queen) {
      for (auto distance = 2; x - distance >= 0 and y + distance < 8;
           distance++)
        if (not state.board.pieces[y + distance][x - distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y + enemy][x - enemy].enemy) {
              auto valid = true;

              for (auto between = enemy + 1; between < distance; between++)
                if (state.board.pieces[y + between][x - between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x - distance, y + distance)});
              }
            }
      for (auto distance = 2; x + distance < 8 and y + distance < 8; distance++)
        if (not state.board.pieces[y + distance][x + distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y + enemy][x + enemy].enemy) {
              auto valid = true;

              for (auto between = enemy + 1; between < distance; between++)
                if (state.board.pieces[y + between][x + between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x + distance, y + distance)});
              }
            }
      for (auto distance = 2; x - distance >= 0 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x - distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y - enemy][x - enemy].enemy) {
              auto valid = true;

              for (auto between = enemy + 1; between < distance; between++)
                if (state.board.pieces[y - between][x - between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x - distance, y - distance)});
              }
            }
      for (auto distance = 2; x + distance < 8 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x + distance].occup)
          for (auto enemy = 1; enemy < distance; enemy++)
            if (state.board.pieces[y - enemy][x + enemy].enemy) {
              auto valid = true;

              for (auto between = enemy + 1; between < distance; between++)
                if (state.board.pieces[y - between][x + between].occup)
                  valid = false;

              if (valid) {
                auto action =
                    (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
                actions.push_back(
                    {action, get_eat_actions(apply_action(state, action),
                                             x + distance, y - distance)});
              }
            }
    } else {
      if (y + 2 < 8) {
        if (x - 2 >= 0 and not state.board.pieces[y + 2][x - 2].occup and
            state.board.pieces[y + 1][x - 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x - 2, y + 2)});
        }
        if (x + 2 < 8 and not state.board.pieces[y + 2][x + 2].occup and
            state.board.pieces[y + 1][x + 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x + 2, y + 2)});
        }
      }
      if (y - 2 >= 0) {
        if (x - 2 >= 0 and not state.board.pieces[y - 2][x - 2].occup and
            state.board.pieces[y - 1][x - 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 2) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x - 2, y - 2)});
        }
        if (x + 2 < 8 and not state.board.pieces[y - 2][x + 2].occup and
            state.board.pieces[y - 1][x + 1].enemy) {
          auto action = (8 * 8 * 4 * 1) + (8 * 8 * 3) + (8 * y) + x;
          actions.push_back(
              {action,
               get_eat_actions(apply_action(state, action), x + 2, y - 2)});
        }
      }
    }

    return actions;
  }

  static constexpr auto get_jump_actions(const State& state, int x, int y)
      -> std::vector<Action> {
    auto actions = std::vector<Action>{};

    auto piece = state.board.pieces[y][x];

    assert(not piece.enemy);

    if (piece.queen) {
      for (auto distance = 1; x - distance >= 0 and y + distance < 8;
           distance++)
        if (not state.board.pieces[y + distance][x - distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y + between][x - between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 0) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x + distance < 8 and y + distance < 8; distance++)
        if (not state.board.pieces[y + distance][x + distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y + between][x + between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 1) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x - distance >= 0 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x - distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y - between][x - between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 2) + (8 * y) + x;
            actions.push_back(action);
          }
        }
      for (auto distance = 1; x + distance < 8 and y - distance >= 0;
           distance++)
        if (not state.board.pieces[y - distance][x + distance].occup) {
          auto valid = true;

          for (auto between = 1; between < distance; between++)
            if (state.board.pieces[y - between][x + between].occup)
              valid = false;

          if (valid) {
            auto action =
                (8 * 8 * 4 * (distance - 1)) + (8 * 8 * 3) + (8 * y) + x;
            actions.push_back(action);
          }
        }
    } else {
      if (y + 1 < 8) {
        if (x - 1 >= 0 and not state.board.pieces[y + 1][x - 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 0) + (8 * y) + x;
          actions.push_back(action);
        }
        if (x + 1 < 8 and not state.board.pieces[y + 1][x + 1].occup) {
          auto action = (8 * 8 * 4 * 0) + (8 * 8 * 1) + (8 * y) + x;
          actions.push_back(action);
        }
      }
    }

    return actions;
  }

  static constexpr auto legal_actions(const State& state) -> torch::Tensor {
    auto legal_actions = std::vector<Action>{};

    auto pieces = std::vector<std::pair<int, int>>{};

    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++)
        if (state.board.pieces[y][x].occup)
          if (not state.board.pieces[y][x].enemy)
            pieces.push_back({x, y});

    auto eat_actions = std::vector<ActionNode>{};
    for (auto& [x, y] : pieces)
      eat_actions.append_range(get_eat_actions(state, x, y));

    auto heights =
        eat_actions |
        std::views::transform([](const auto& node) { return node.height(); }) |
        std::ranges::to<std::vector>();

    for (auto it = std::max_element(heights.begin(), heights.end());
         it != heights.end(); it = std::find(it + 1, heights.end(), *it))
      legal_actions.push_back(
          eat_actions[std::distance(heights.begin(), it)].action);

    auto legal_actions_tensor = torch::zeros(ActionSize, torch::kFloat32);
    if (not legal_actions.empty()) {
      for (auto& action : legal_actions)
        legal_actions_tensor[action] = 1.0;

      return legal_actions_tensor;
    }

    for (auto& [x, y] : pieces)
      legal_actions.append_range(get_jump_actions(state, x, y));

    for (auto& action : legal_actions)
      legal_actions_tensor[action] = 1.0;

    return legal_actions_tensor;
  }

  static constexpr auto get_outcome(const State& state, Action action)
      -> std::optional<az::GameOutcome> {
    if (state.draw_count >= 80)
      return az::GameOutcome::Draw;

    if (legal_actions(state).sum(0).item<double>() == 0.0) {
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
      canonical_state.board = flip(state.board);
      canonical_state.player = state.player.next();
      canonical_state.scores.first = state.scores.second;
      canonical_state.scores.second = state.scores.first;

      auto piece = canonical_state.board.pieces[new_y][new_x];

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

  static constexpr auto encode_state(const State& state) -> torch::Tensor {
    auto encoded_state = torch::zeros({8, 8, 6}, torch::kFloat32);
    for (int x = 0; x < 8; x++) {
      for (int y = 0; y < 8; y++) {
        if (state.player.is_first()) {
          encoded_state[x][y][0] = state.scores.first;
          encoded_state[x][y][1] = state.scores.second;
          if (state.board.pieces[y][x].occup) {
            auto& piece = state.board.pieces[y][x];
            auto value = (piece.ngtve ? -1 : 1) * piece.value;
            encoded_state[x][y][(not piece.enemy ? 2 : 4) +
                                (piece.queen ? 1 : 0)] = value;
          }
        } else {
          encoded_state[x][y][0] = state.scores.second;
          encoded_state[x][y][1] = state.scores.first;
          if (state.board.pieces[y][x].occup) {
            auto& piece = state.board.pieces[y][x];
            auto value = (piece.ngtve ? -1 : 1) * piece.value;
            encoded_state[x][y][(piece.enemy ? 2 : 4) + (piece.queen ? 1 : 0)] =
                value;
          }
        }
      }
    }

    return encoded_state;
  }

  static constexpr auto print(const State& state) -> void {
    auto& board = state.board;
    auto [first, second] = state.scores;
    std::print("Score: {:.5f} {:.5f}\n", first, second);
    for (int i = 7; i >= 0; i--) {
      std::print(" {: ^3} ", i);
      for (int j = 0; j < 8; j++) {
        if (board.pieces[i][j].occup) {
          if (board.pieces[i][j].queen)
            std::cout << "\033[1m";  // bold
          std::cout << (board.pieces[i][j].enemy ? "\033[31m"
                                                 : "\033[34m");  // blue
          std::print(" {: ^3} ", (board.pieces[i][j].ngtve ? -1 : 1) *
                                     board.pieces[i][j].value);
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